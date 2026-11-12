#include <string>

namespace Sapphire::Graphics::Shaders {
	inline const std::string deformationShaderSource = R"(
		#include <metal_stdlib>
		using namespace metal;
		float3 InstanceRotate(float3 point, float3 degrees) {
			float3 angle = degrees * 0.01745329252;
			float3 cosine = cos(angle);
			float3 sine = sin(angle);
			float3 axisX = float3(cosine.z * cosine.y, sine.z * cosine.y, -sine.y);
			float3 axisY = float3(cosine.z * sine.y * sine.x - sine.z * cosine.x, sine.z * sine.y * sine.x + cosine.z * cosine.x, cosine.y * sine.x);
			float3 axisZ = float3(cosine.z * sine.y * cosine.x + sine.z * sine.x, sine.z * sine.y * cosine.x - cosine.z * sine.x, cosine.y * cosine.x);
			return axisX * point.x + axisY * point.y + axisZ * point.z;
		}
		float3 InstanceNormal(float3 normal, float3 scale, float3 rotation) {
			float3 inverseScale = select(float3(0.0), 1.0 / scale, abs(scale) > float3(0.0001));
			float3 result = InstanceRotate(normal * inverseScale, rotation);
			float lengthSquared = dot(result, result);
			return lengthSquared > 0.0 ? result * rsqrt(lengthSquared) : float3(0.0);
		}
		float3 Deform(float3 position, float3 normal, float2 uv, float4 color, float4x4 model, float4 parameters, texture2d<float> heightMap, sampler heightSampler, texture2d<float> maskMap, sampler maskSampler) {
			if (parameters.x != 0.0) position += normal * ((heightMap.sample(heightSampler, uv, level(0.0)).r - 0.5) * maskMap.sample(maskSampler, uv, level(0.0)).r * parameters.x);
			return position;
		}
		#ifndef CUSTOM_MODIFIER
		void ModifyVertex(thread float3& position, thread float3& normal, thread float4& color, float instanceData, float time, float4x4 model) {}
		#endif
	)";
	
	inline const std::string vertexShaderSource = deformationShaderSource + R"(
		struct VertexInput { float3 position [[attribute(0)]]; float3 normal [[attribute(1)]]; float2 uv [[attribute(2)]]; float4 color [[attribute(3)]];
		#if SKINNED
		ushort4 boneIndices [[attribute(4)]]; float4 boneWeights [[attribute(5)]];
		#endif
		#if INSTANCED
		float3 instancePosition [[attribute(4)]]; float3 instanceRotation [[attribute(5)]]; float3 instanceScale [[attribute(6)]]; float instanceData [[attribute(7)]];
		#endif
		};
		struct VertexOutput { float4 position [[position]]; float2 uv; float4 color; float3 worldPosition; float3 worldNormal; float4 shadowPosition0; float4 shadowPosition1; float4 shadowPosition2; };
		struct ObjectUniforms { float4x4 mvp; float4x4 model; float4x4 normal; float4 displacementParameters; float4 uvTransform; float4 modifierTime; float4x4 shadowMvp[3]; float4 shadowState; };
		vertex VertexOutput main0(VertexInput input [[stage_in]], constant ObjectUniforms& uniforms [[buffer(0)]],
		#if SKINNED
		device const float4x4* boneMatrices [[buffer(1)]],
		#endif
		texture2d<float> heightMap [[texture(0)]], sampler heightSampler [[sampler(0)]], texture2d<float> maskMap [[texture(1)]], sampler maskSampler [[sampler(1)]]) {
			VertexOutput output;
			float2 uv = input.uv * uniforms.uvTransform.xy + uniforms.uvTransform.zw;
			float3 localPosition = input.position;
			float3 localNormal = input.normal;
			float4 color = input.color;
			#if SKINNED
			float4 skinnedPosition = float4(0.0);
			float3 skinnedNormal = float3(0.0);
			for (uint influence = 0; influence < 4; ++influence) {
				float weight = input.boneWeights[influence];
				skinnedPosition += (boneMatrices[input.boneIndices[influence]] * float4(localPosition, 1.0)) * weight;
				skinnedNormal += (boneMatrices[input.boneIndices[influence]] * float4(localNormal, 0.0)).xyz * weight;
			}
			localPosition = skinnedPosition.xyz;
			localNormal = normalize(skinnedNormal);
			#endif
			#if INSTANCED
			localPosition = input.instancePosition + InstanceRotate(localPosition * input.instanceScale, input.instanceRotation);
			localNormal = InstanceNormal(localNormal, input.instanceScale, input.instanceRotation);
			#endif
			float3 position = Deform(localPosition, localNormal, uv, color, uniforms.model, uniforms.displacementParameters, heightMap, heightSampler, maskMap, maskSampler);
			float instanceData = 0.0;
			#if INSTANCED
			instanceData = input.instanceData;
			#endif
			ModifyVertex(position, localNormal, color, instanceData, uniforms.modifierTime.x, uniforms.model);
			output.position = uniforms.mvp * float4(position, 1.0);
			output.uv = uv;
			output.color = color;
			output.worldPosition = (uniforms.model * float4(position, 1.0)).xyz;
			output.worldNormal = normalize((uniforms.normal * float4(localNormal, 0.0)).xyz);
			output.shadowPosition0 = output.shadowPosition1 = output.shadowPosition2 = float4(0.0);
			if (uniforms.shadowState.x > 0.5) {
				output.shadowPosition0 = uniforms.shadowMvp[0] * float4(output.worldPosition, 1.0);
				output.shadowPosition1 = uniforms.shadowMvp[1] * float4(output.worldPosition, 1.0);
				output.shadowPosition2 = uniforms.shadowMvp[2] * float4(output.worldPosition, 1.0);
			}
			return output;
		}
	)";

	inline const std::string fogCommonShaderSource = R"(
		#include <metal_stdlib>
		using namespace metal;
		struct FrameUniforms {
			float4 cameraPosition; float4 fogColor; float4 fogParameters; float4 fogHeightParameters;
			float4 fogVolumeParameters; float4 fogSliceParameters; float4 fogScreenParameters; float4 fogViewParameters;
			float4 cameraRight; float4 cameraUp; float4 cameraForward;
			float4 ambientLight; float4 directionalDirection; float4 directionalColor;
			float4 pointLightPositionRange[16]; float4 pointLightColorIntensity[16];
			float4 pointLightShadowInfo[16]; float4 pointLightShadowBias[16];
		};
		float HeightFogProfile(float3 position, constant FrameUniforms& frame, float4 profile) {
			float3 ray = position - frame.cameraPosition.xyz;
			float rayLength = length(ray);
			float endDistance = min(rayLength, profile.w);
			float verticalDirection = ray.z / max(rayLength, 0.000001);
			float startHeight = frame.cameraPosition.z;
			float endHeight = frame.cameraPosition.z + verticalDirection * endDistance;
			float falloff = profile.y;
			float density = profile.x * exp2(clamp(-falloff * (min(startHeight, endHeight) - profile.z), -80.0, 80.0));
			float heightChange = falloff * abs(endHeight - startHeight) * 0.69314718056;
			float integral = heightChange < 0.01 ? 1.0 - heightChange * 0.5 + heightChange * heightChange / 6.0 : (1.0 - exp2(-heightChange * 1.44269504089)) / heightChange;
			float distance = min(rayLength / profile.w, 1.0);
			float distanceSquared = distance * distance;
			float distanceFourth = distanceSquared * distanceSquared;
			float opticalDepth = min(density * endDistance * integral + 4.6 * distanceFourth * distanceFourth, 80.0);
			return 1.0 - exp2(-opticalDepth * 1.44269504089);
		}
		float HeightFog(float3 position, constant FrameUniforms& frame) {
			if (frame.fogParameters.z < 0.5) return 0.0;
			return HeightFogProfile(position, frame, float4(frame.fogHeightParameters.x, frame.fogHeightParameters.y, frame.fogHeightParameters.z, frame.fogParameters.y));
		}
	)";

	inline const std::string fogComputeShaderSource = fogCommonShaderSource + R"(
		struct FogDensityInputUniforms { float4 boundsMinimum; float4 boundsMaximum; float4 parameters; float4 grid; };
		struct FogComputeUniforms { FrameUniforms frame; uint4 densityInputCount; FogDensityInputUniforms densityInputs[5]; };
		kernel void main0(uint2 gid [[thread_position_in_grid]], constant FogComputeUniforms& uniforms [[buffer(0)]], texture3d<float> densityMap0 [[texture(0)]], sampler densitySampler0 [[sampler(0)]], texture3d<float> densityMap1 [[texture(1)]], sampler densitySampler1 [[sampler(1)]], texture3d<float> densityMap2 [[texture(2)]], sampler densitySampler2 [[sampler(2)]], texture3d<float> densityMap3 [[texture(3)]], sampler densitySampler3 [[sampler(3)]], texture3d<float, access::write> fogOutput [[texture(4)]]) {
			constant FrameUniforms& frame = uniforms.frame;
			uint2 gridSize = uint2(frame.fogVolumeParameters.zw);
			if (gid.x >= gridSize.x || gid.y >= gridSize.y) return;
			float2 uv = (float2(gid) + 0.5) / float2(gridSize);
			float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
			float3 ray = normalize(frame.cameraForward.xyz + frame.cameraRight.xyz * ndc.x * frame.fogViewParameters.x * frame.fogViewParameters.y + frame.cameraUp.xyz * ndc.y * frame.fogViewParameters.x);
			float nearClip = frame.fogViewParameters.z;
			float farClip = frame.fogViewParameters.w;
			uint sliceCount = max(uint(frame.fogSliceParameters.x), 1u);
			float logRange = log(farClip / nearClip);
			float previousDistance = nearClip;
			float previousFog[5] = {0.0};
			float transmittance = 1.0;
			float3 scattering = float3(0.0);
			for (uint slice = 0; slice < sliceCount; ++slice) {
				float distance = nearClip * exp(logRange * (float(slice) + 0.5) / float(sliceCount));
				float3 position = frame.cameraPosition.xyz + ray * distance;
				float segmentTransmittance = 1.0;
				for (uint sourceIndex = 0; sourceIndex <= uniforms.densityInputCount.x; ++sourceIndex) {
					FogDensityInputUniforms source = uniforms.densityInputs[sourceIndex];
					if (source.parameters.x < 0.5) {
						float fog = HeightFogProfile(position, frame, float4(source.parameters.y, source.parameters.z, source.parameters.w, source.grid.w));
						float previousTransmittance = 1.0 - previousFog[sourceIndex];
						segmentTransmittance *= clamp((1.0 - fog) / max(previousTransmittance, 0.000001), 0.0, 1.0);
						previousFog[sourceIndex] = fog;
					}
					else {
						float3 extent = source.boundsMaximum.xyz - source.boundsMinimum.xyz;
						float3 coordinate = (position - source.boundsMinimum.xyz) / max(extent, float3(0.000001));
						if (all(coordinate >= float3(0.0)) && all(coordinate <= float3(1.0))) {
							float density = 0.0;
							if (sourceIndex == 1) density = densityMap0.sample(densitySampler0, coordinate).r;
							else if (sourceIndex == 2) density = densityMap1.sample(densitySampler1, coordinate).r;
							else if (sourceIndex == 3) density = densityMap2.sample(densitySampler2, coordinate).r;
							else if (sourceIndex == 4) density = densityMap3.sample(densitySampler3, coordinate).r;
							segmentTransmittance *= exp(-max(density, 0.0) * max(distance - previousDistance, 0.0));
						}
					}
				}
				segmentTransmittance = clamp(segmentTransmittance, 0.0, 1.0);
				float segmentScattering = transmittance * (1.0 - segmentTransmittance) * frame.fogVolumeParameters.y;
				float3 lighting = frame.ambientLight.rgb * frame.ambientLight.a;
				float directionalIntensity = frame.directionalDirection.w;
				if (directionalIntensity > 0.0) lighting += frame.directionalColor.rgb * directionalIntensity;
				uint lightCount = min(uint(frame.fogParameters.x), 16u);
				for (uint light = 0; light < lightCount; ++light) {
					float4 positionRange = frame.pointLightPositionRange[light];
					float4 colorIntensity = frame.pointLightColorIntensity[light];
					if (positionRange.w <= 0.0 || colorIntensity.w <= 0.0) continue;
					float3 toLight = positionRange.xyz - position;
					float distanceSquared = dot(toLight, toLight);
					if (distanceSquared >= positionRange.w * positionRange.w) continue;
					float rangeFade = clamp(1.0 - sqrt(distanceSquared) / positionRange.w, 0.0, 1.0);
					lighting += colorIntensity.rgb * colorIntensity.w * rangeFade * rangeFade / max(distanceSquared, 1.0);
				}
				scattering += frame.fogColor.rgb * lighting * segmentScattering;
				transmittance *= segmentTransmittance;
				previousDistance = distance;
				fogOutput.write(float4(scattering, transmittance), uint3(gid, slice));
			}
		}
	)";

	inline const std::string fogBackgroundVertexShaderSource = R"(
		#include <metal_stdlib>
		using namespace metal;
		struct VertexOutput { float4 position [[position]]; };
		vertex VertexOutput main0(uint vertexId [[vertex_id]]) {
			float2 positions[3] = {float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0)};
			VertexOutput output;
			output.position = float4(positions[vertexId], 1.0, 1.0);
			return output;
		}
	)";

	inline const std::string fogBackgroundFragmentShaderSource = fogCommonShaderSource + R"(
		struct BackgroundUniforms { float4 color; };
		fragment float4 main0(float4 position [[position]], constant BackgroundUniforms& background [[buffer(0)]], constant FrameUniforms& frame [[buffer(1)]], texture3d<float> fogVolumeMap [[texture(0)]], sampler fogVolumeSampler [[sampler(0)]]) {
			float2 screenUV = position.xy / frame.fogScreenParameters.xy;
			float sliceCount = max(frame.fogSliceParameters.x, 1.0);
			float3 fogCoordinate = float3(screenUV, (sliceCount - 0.5) / sliceCount);
			float4 integratedFog = fogVolumeMap.sample(fogVolumeSampler, fogCoordinate);
			float3 color = background.color.rgb * integratedFog.a + integratedFog.rgb;
			color = max(color, float3(0.0));
			if (frame.fogHeightParameters.w < 0.5) color = select(1.055 * pow(color, float3(1.0 / 2.4)) - 0.055, color * 12.92, color <= 0.0031308);
			return float4(color, 1.0);
		}
	)";

	inline const std::string fragmentShaderSource = fogCommonShaderSource + R"(
		struct FragmentInput { float4 position [[position]]; float2 uv; float4 color; float3 worldPosition; float3 worldNormal; float4 shadowPosition0; float4 shadowPosition1; float4 shadowPosition2; };
		struct FragmentUniforms { float4 shadowParameters[3]; float4 lightingParameters; float4 baseColor; float4 surfaceParameters; uint4 pointLightIndices[4]; };
		float ShadowCoverage(float4 position, float4 parameters) {
			if (parameters.x < 0.5) return 0.0;
			float3 projected = position.xyz / position.w;
			float2 uv = float2(projected.x * 0.5 + 0.5, 0.5 - projected.y * 0.5);
			float border = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
			if (projected.z < 0.0 || (projected.z > 1.0 && parameters.w < 0.5)) return 0.0;
			return smoothstep(parameters.z * 2.0, parameters.z * 6.0, border);
		}
		float ShadowAmount(float4 position, depth2d<float> shadowMap, sampler shadowSampler, float4 parameters) {
			if (ShadowCoverage(position, parameters) == 0.0) return 0.0;
			float3 projected = position.xyz / position.w;
			float2 uv = float2(projected.x * 0.5 + 0.5, 0.5 - projected.y * 0.5);
			float visibility = 0.0;
			for (int y = 0; y < 2; ++y) {
				for (int x = 0; x < 2; ++x) visibility += shadowMap.sample_compare(shadowSampler, uv + (float2(x, y) - 0.5) * parameters.z, min(projected.z, 1.0) - parameters.y);
			}
			return 1.0 - visibility * 0.25;
		}
		float PointShadowVisibility(float3 direction, float receiverDistance, float range, float bias, uint cubeIndex, depth2d_array<float> shadowMap, sampler shadowSampler) {
			float faceDepth = max(max(abs(direction.x), abs(direction.y)), abs(direction.z)) * receiverDistance;
			float nearClip = min(0.1, range * 0.01);
			float compareDistance = max(faceDepth - bias, nearClip);
			float compareDepth = range / (range - nearClip) - nearClip * range / ((range - nearClip) * compareDistance);
			float3 magnitude = abs(direction);
			float2 uv;
			uint face;
			if (magnitude.x >= magnitude.y && magnitude.x >= magnitude.z) {
				face = direction.x > 0.0 ? 0u : 1u;
				uv = float2(direction.x > 0.0 ? -direction.z : direction.z, -direction.y) / magnitude.x;
			}
			else if (magnitude.y >= magnitude.z) {
				face = direction.y > 0.0 ? 2u : 3u;
				uv = float2(direction.x, direction.y > 0.0 ? direction.z : -direction.z) / magnitude.y;
			}
			else {
				face = direction.z > 0.0 ? 4u : 5u;
				uv = float2(direction.z > 0.0 ? direction.x : -direction.x, -direction.y) / magnitude.z;
			}
			return shadowMap.sample_compare(shadowSampler, uv * 0.5 + 0.5, cubeIndex * 6u + face, compareDepth);
		}
		float PointShadowAmount(float3 direction, float range, float bias, uint cubeIndex, depth2d_array<float> shadowMap, sampler shadowSampler, float texelSize) {
			float receiverDistance = length(direction);
			if (receiverDistance <= 0.0001 || receiverDistance >= range) return 0.0;
			float3 normalizedDirection = direction / receiverDistance;
			float3 axis = abs(normalizedDirection.y) < 0.99 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
			float3 tangent = normalize(cross(normalizedDirection, axis));
			float3 bitangent = cross(normalizedDirection, tangent);
			float filterSize = texelSize * 2.0;
			float visibility = 0.0;
			visibility += PointShadowVisibility(normalize(normalizedDirection + (tangent + bitangent) * filterSize), receiverDistance, range, bias, cubeIndex, shadowMap, shadowSampler);
			visibility += PointShadowVisibility(normalize(normalizedDirection + (tangent - bitangent) * filterSize), receiverDistance, range, bias, cubeIndex, shadowMap, shadowSampler);
			visibility += PointShadowVisibility(normalize(normalizedDirection + (-tangent + bitangent) * filterSize), receiverDistance, range, bias, cubeIndex, shadowMap, shadowSampler);
			visibility += PointShadowVisibility(normalize(normalizedDirection - (tangent + bitangent) * filterSize), receiverDistance, range, bias, cubeIndex, shadowMap, shadowSampler);
			return 1.0 - visibility * 0.25;
		}

		struct Surface {
			float3 normal;
			float3 view;
			float3 albedo;
			float3 f0;
			float metallic;
			float alphaSquared;
			float nDotV;
			float viewVisibility;
		};
		float3 EvaluateLight(Surface surface, float3 light, bool twoSided) {
			float nDotL = max(twoSided ? abs(dot(surface.normal, light)) : dot(surface.normal, light), 0.0);
			if (nDotL <= 0.0) return float3(0.0);
			float3 halfVector = surface.view + light;
			halfVector *= rsqrt(max(dot(halfVector, halfVector), 0.000001));
			float nDotH = clamp(twoSided ? abs(dot(surface.normal, halfVector)) : dot(surface.normal, halfVector), 0.0, 1.0);
			float vDotH = clamp(dot(surface.view, halfVector), 0.0, 1.0);
			float denominator = nDotH * nDotH * (surface.alphaSquared - 1.0) + 1.0;
			float distribution = surface.alphaSquared / max(3.14159265 * denominator * denominator, 0.000000000001);
			float visibility = 0.5 / max(nDotL * surface.viewVisibility + surface.nDotV * sqrt(nDotL * nDotL * (1.0 - surface.alphaSquared) + surface.alphaSquared), 0.00000001);
			float fresnelFactor = 1.0 - vDotH;
			float fresnelSquared = fresnelFactor * fresnelFactor;
			float fresnelWeight = fresnelSquared * fresnelSquared * fresnelFactor;
			float3 fresnel = surface.f0 + (1.0 - surface.f0) * fresnelWeight;
			float3 diffuse = (1.0 - fresnel) * (1.0 - surface.metallic) * surface.albedo / 3.14159265;
			return (diffuse + distribution * visibility * fresnel) * nDotL;
		}

		float3 EncodeColor(float3 color, constant FrameUniforms& frame) {
			if (frame.fogHeightParameters.w > 0.5) return color;
			color = max(color, float3(0.0));
			return select(1.055 * pow(color, float3(1.0 / 2.4)) - 0.055, color * 12.92, color <= 0.0031308);
		}
		fragment float4 main0(FragmentInput input [[stage_in]], constant FragmentUniforms& uniforms [[buffer(0)]], constant FrameUniforms& frame [[buffer(1)]], texture2d<float> image [[texture(0)]], sampler imageSampler [[sampler(0)]], depth2d<float> detailMap [[texture(1)]], sampler detailSampler [[sampler(1)]], depth2d<float> fallbackMap [[texture(2)]], sampler fallbackSampler [[sampler(2)]], depth2d<float> dynamicMap [[texture(3)]], sampler dynamicSampler [[sampler(3)]], depth2d_array<float> pointShadowMap [[texture(4)]], sampler pointShadowSampler [[sampler(4)]], texture2d<float> normalMap [[texture(5)]], sampler normalSampler [[sampler(5)]], texture2d<float> roughnessMap [[texture(6)]], sampler roughnessSampler [[sampler(6)]], texture2d<float> aoMap [[texture(7)]], sampler aoSampler [[sampler(7)]], texture3d<float> fogVolumeMap [[texture(8)]], sampler fogVolumeSampler [[sampler(8)]]) {
			uint materialFlags = uint(uniforms.lightingParameters.x);
			bool twoSided = (materialFlags & 32u) != 0u;
			float3 normal = (materialFlags & 16u) != 0u ? float3(0.0) : normalize(input.worldNormal);
			if ((materialFlags & 16u) == 0u && (materialFlags & 1u) != 0u) {
				float3 dpdx = dfdx(input.worldPosition);
				float3 dpdy = dfdy(input.worldPosition);
				float2 duvdx = dfdx(input.uv);
				float2 duvdy = dfdy(input.uv);
				float determinant = duvdx.x * duvdy.y - duvdx.y * duvdy.x;
				if (determinant * determinant > 1e-8 * dot(duvdx, duvdx) * dot(duvdy, duvdy)) {
					float3 tangent = (dpdx * duvdy.y - dpdy * duvdx.y) / determinant;
					float3 uvBitangent = (dpdy * duvdx.x - dpdx * duvdy.x) / determinant;
					tangent -= normal * dot(normal, tangent);
					tangent *= rsqrt(max(dot(tangent, tangent), 1e-20));
					float3 bitangent = cross(normal, tangent);
					if (dot(bitangent, uvBitangent) < 0.0) bitangent = -bitangent;
					float3 mappedNormal = normalMap.sample(normalSampler, input.uv).xyz * 2.0 - 1.0;
					mappedNormal.xy *= float2(uniforms.surfaceParameters.y, uniforms.surfaceParameters.y * uniforms.surfaceParameters.w);
					normal = normalize(tangent * mappedNormal.x + bitangent * mappedNormal.y + normal * mappedNormal.z);
				}
			}
			float4 texel = image.sample(imageSampler, input.uv);
			#if ALPHA_MASKED
			if (texel.a < 0.5) discard_fragment();
			#endif
			float fog = HeightFog(input.worldPosition, frame);
			float3 fogScattering = float3(0.0);
			float fogTransmittance = 1.0 - fog;
			if (frame.fogVolumeParameters.x > 0.5) {
				float nearClip = frame.fogViewParameters.z;
				float farClip = frame.fogViewParameters.w;
				float rayLength = length(input.worldPosition - frame.cameraPosition.xyz);
				float depth = clamp(log(max(rayLength, nearClip) / nearClip) / log(farClip / nearClip), 0.0, 1.0);
				float sliceCount = max(frame.fogSliceParameters.x, 1.0);
				float sliceCoordinate = depth * ((sliceCount - 1.0) / sliceCount) + (0.5 / sliceCount);
				float2 screenUV = input.position.xy / frame.fogScreenParameters.xy;
				float4 integratedFog = fogVolumeMap.sample(fogVolumeSampler, float3(screenUV, sliceCoordinate));
				fogScattering = integratedFog.rgb;
				fogTransmittance = integratedFog.a;
			}
			float opacity = input.color.a * texel.a * uniforms.lightingParameters.y;
			if (fog >= 1.0 && frame.fogVolumeParameters.x < 0.5) return float4(EncodeColor(frame.fogColor.rgb, frame), opacity);
			if ((materialFlags & 16u) != 0u) {
				float3 tint = input.color.rgb * uniforms.baseColor.rgb * texel.rgb;
				float3 color = frame.fogVolumeParameters.x > 0.5 ? tint * fogTransmittance + fogScattering : mix(tint, frame.fogColor.rgb, fog);
				return float4(EncodeColor(color, frame), opacity);
			}
			float3 albedo = input.color.rgb * texel.rgb * uniforms.baseColor.rgb;
			float roughness = clamp(uniforms.surfaceParameters.x * ((materialFlags & 2u) != 0u ? roughnessMap.sample(roughnessSampler, input.uv).r : 1.0), 0.045, 1.0);
			float ao = (materialFlags & 4u) != 0u ? mix(1.0, aoMap.sample(aoSampler, input.uv).r, uniforms.surfaceParameters.z) : 1.0;
			float metallic = uniforms.baseColor.w;
			float3 view = frame.cameraPosition.xyz - input.worldPosition;
			view *= rsqrt(max(dot(view, view), 0.000001));
			float alphaSquared = roughness * roughness * roughness * roughness;
			float nDotV = max(twoSided ? abs(dot(normal, view)) : dot(normal, view), 0.0001);
			Surface surface{normal, view, albedo, mix(float3(0.04), albedo, metallic), metallic, alphaSquared, nDotV, sqrt(nDotV * nDotV * (1.0 - alphaSquared) + alphaSquared)};
			float diffuse = max(twoSided ? abs(dot(normal, -frame.directionalDirection.xyz)) : dot(normal, -frame.directionalDirection.xyz), 0.0) * frame.directionalDirection.w;
			bool directionalShadow = uniforms.lightingParameters.z > 0.5 && diffuse > 0.0;
			float staticShadow = 0.0;
			if (directionalShadow) {
				float detailCoverage = ShadowCoverage(input.shadowPosition0, uniforms.shadowParameters[0]);
				if (detailCoverage < 1.0) staticShadow = ShadowAmount(input.shadowPosition1, fallbackMap, fallbackSampler, uniforms.shadowParameters[1]);
				if (detailCoverage > 0.0) staticShadow = mix(staticShadow, ShadowAmount(input.shadowPosition0, detailMap, detailSampler, uniforms.shadowParameters[0]), detailCoverage);
			}
			float dynamicShadow = 0.0;
			if (directionalShadow) dynamicShadow = ShadowAmount(input.shadowPosition2, dynamicMap, dynamicSampler, uniforms.shadowParameters[2]);
			float3 direct = frame.directionalDirection.w > 0.0 ? EvaluateLight(surface, -frame.directionalDirection.xyz, twoSided) * frame.directionalColor.rgb * frame.directionalDirection.w * (1.0 - staticShadow) * (1.0 - dynamicShadow) : float3(0.0);
			float3 pointLighting = float3(0.0);
			for (int light = 0; light < int(uniforms.lightingParameters.w); ++light) {
				uint index = uniforms.pointLightIndices[light / 4][light % 4];
				float4 positionRange = frame.pointLightPositionRange[index];
				float4 colorIntensity = frame.pointLightColorIntensity[index];
				float3 toLight = positionRange.xyz - input.worldPosition;
				float distanceSquared = dot(toLight, toLight);
				if (distanceSquared < positionRange.w * positionRange.w) {
					float distanceToLight = sqrt(distanceSquared);
					float pointDiffuse = max(twoSided ? abs(dot(normal, toLight / max(distanceToLight, 0.0001))) : dot(normal, toLight / max(distanceToLight, 0.0001)), 0.0);
					if (pointDiffuse <= 0.0) continue;
					float rangeFade = clamp(1.0 - distanceToLight / positionRange.w, 0.0, 1.0);
					float attenuation = rangeFade * rangeFade / max(distanceSquared, 1.0);
					float4 shadowInfo = frame.pointLightShadowInfo[index];
					float shadowBias = frame.pointLightShadowBias[index].x;
					float staticPointShadow = uniforms.lightingParameters.z > 0.5 && shadowInfo.z > 0.5 ? PointShadowAmount(-toLight, positionRange.w, shadowBias, uint(shadowInfo.x), pointShadowMap, pointShadowSampler, uniforms.shadowParameters[0].z) : 0.0;
					float dynamicPointShadow = uniforms.lightingParameters.z > 0.5 && shadowInfo.w > 0.5 ? PointShadowAmount(-toLight, positionRange.w, shadowBias, uint(shadowInfo.y), pointShadowMap, pointShadowSampler, uniforms.shadowParameters[0].z) : 0.0;
					pointLighting += EvaluateLight(surface, toLight / max(distanceToLight, 0.0001), twoSided) * colorIntensity.rgb * colorIntensity.a * attenuation * (1.0 - staticPointShadow) * (1.0 - dynamicPointShadow);
				}
			}
			float3 ambient = frame.ambientLight.a > 0.5 ? frame.ambientLight.rgb : float3(0.0);
			float3 color = albedo * ambient * ao * (1.0 - metallic) + direct + pointLighting;
			color = frame.fogVolumeParameters.x > 0.5 ? color * fogTransmittance + fogScattering : mix(color, frame.fogColor.rgb, fog);
			return float4(EncodeColor(color, frame), opacity);
		}
	)";

	inline const std::string shadowVertexShaderSource = deformationShaderSource + R"(
		struct VertexInput { float3 position [[attribute(0)]]; float3 normal [[attribute(1)]]; float2 uv [[attribute(2)]]; float4 color [[attribute(3)]];
		#if SKINNED
		ushort4 boneIndices [[attribute(4)]]; float4 boneWeights [[attribute(5)]];
		#endif
		#if INSTANCED
		float3 instancePosition [[attribute(4)]]; float3 instanceRotation [[attribute(5)]]; float3 instanceScale [[attribute(6)]]; float instanceData [[attribute(7)]];
		#endif
		};
		struct VertexOutput { float4 position [[position]]; float2 uv; };
		struct ShadowUniforms { float4x4 mvp; float4x4 model; float4 displacementParameters; float4 uvTransform; float4 modifierTime; };
		vertex VertexOutput main0(VertexInput input [[stage_in]], constant ShadowUniforms& uniforms [[buffer(0)]],
		#if SKINNED
		device const float4x4* boneMatrices [[buffer(1)]],
		#endif
		texture2d<float> heightMap [[texture(0)]], sampler heightSampler [[sampler(0)]], texture2d<float> maskMap [[texture(1)]], sampler maskSampler [[sampler(1)]]) {
			VertexOutput output;
			float2 uv = input.uv * uniforms.uvTransform.xy + uniforms.uvTransform.zw;
			float3 localPosition = input.position;
			float3 localNormal = input.normal;
			float4 color = input.color;
			#if SKINNED
			float4 skinnedPosition = float4(0.0);
			float3 skinnedNormal = float3(0.0);
			for (uint influence = 0; influence < 4; ++influence) {
				float weight = input.boneWeights[influence];
				skinnedPosition += (boneMatrices[input.boneIndices[influence]] * float4(localPosition, 1.0)) * weight;
				skinnedNormal += (boneMatrices[input.boneIndices[influence]] * float4(localNormal, 0.0)).xyz * weight;
			}
			localPosition = skinnedPosition.xyz;
			localNormal = normalize(skinnedNormal);
			#endif
			#if INSTANCED
			localPosition = input.instancePosition + InstanceRotate(localPosition * input.instanceScale, input.instanceRotation);
			localNormal = InstanceNormal(localNormal, input.instanceScale, input.instanceRotation);
			#endif
			float3 position = Deform(localPosition, localNormal, uv, color, uniforms.model, uniforms.displacementParameters, heightMap, heightSampler, maskMap, maskSampler);
			float instanceData = 0.0;
			#if INSTANCED
			instanceData = input.instanceData;
			#endif
			ModifyVertex(position, localNormal, color, instanceData, uniforms.modifierTime.x, uniforms.model);
			output.position = uniforms.mvp * float4(position, 1.0);
			output.uv = uv;
			return output;
		}
	)";

	inline constexpr const char* shadowFragmentShaderSource = R"(
		#include <metal_stdlib>
		using namespace metal;
		struct FragmentInput { float4 position [[position]]; float2 uv; };
		#if ALPHA_MASKED
		fragment void main0(FragmentInput input [[stage_in]], texture2d<float> image [[texture(0)]], sampler imageSampler [[sampler(0)]]) {
			if (image.sample(imageSampler, input.uv).a < 0.5) discard_fragment();
		}
		#else
		fragment void main0() {}
		#endif
	)";

	inline constexpr const char* vertexSource = R"(using namespace metal;
	struct Input { float3 position [[attribute(0)]]; };
	struct Output { float4 position [[position]]; };
	vertex Output main0(Input input [[stage_in]], constant float4x4& mvp [[buffer(0)]]) { return {mvp * float4(input.position, 1.0)}; }
	)";

	inline constexpr const char* fragmentSource = R"(using namespace metal;
	fragment float4 main0() { return float4(0.0, 1.0, 1.0, 1.0); }
	)";

	inline constexpr const char* guiVertexShaderSource = R"(
		#include <metal_stdlib>
		using namespace metal;
		struct VertexInput { float2 position [[attribute(0)]]; float2 uv [[attribute(1)]]; };
		struct VertexOutput { float4 position [[position]]; float2 uv; };
		vertex VertexOutput main0(VertexInput input [[stage_in]], constant float2& scale [[buffer(0)]]) {
			VertexOutput output;
			output.position = float4(input.position * scale + float2(-1.0, 1.0), 0.0, 1.0);
			output.uv = input.uv;
			return output;
		}
	)";

	inline constexpr const char* guiFragmentShaderSource = R"(
		#include <metal_stdlib>
		using namespace metal;
		struct FragmentInput { float4 position [[position]]; float2 uv; };
		fragment float4 main0(FragmentInput input [[stage_in]],
			texture2d<float> image [[texture(0)]], sampler imageSampler [[sampler(0)]]) {
			return image.sample(imageSampler, input.uv);
		}
	)";
}
