#pragma once
#include <SDL3/SDL_gpu.h>
#include <cstring>

namespace Sapphire::Graphics {
    struct GPUBackend { const char* driver; SDL_GPUShaderFormat shaderFormat; };
    inline constexpr GPUBackend gpuBackend{"metal", SDL_GPU_SHADERFORMAT_MSL};
    struct ShaderCode { const Uint8* bytes = nullptr; std::size_t size = 0; };
    static ShaderCode PrepareShaderCode(const char* source, const wchar_t*) {
        return {reinterpret_cast<const Uint8*>(source), std::strlen(source) + 1};
    }
    static Uint32 VisibilitySamplerCount(Uint32 slot, bool singleLayer) { return slot + singleLayer; }
    static Uint32 VisibilityStorageCount(bool) { return 0; }
    static SDL_GPUTextureUsageFlags VisibilityTextureUsage() { return SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER; }
    static void BindVisibilityTexture(SDL_GPURenderPass* pass, SDL_GPUTexture* texture, SDL_GPUSampler* sampler, bool hasDepthSurface) {
        const SDL_GPUTextureSamplerBinding binding{texture, sampler};
        SDL_BindGPUFragmentSamplers(pass, hasDepthSurface ? 10 : 9, &binding, 1);
    }
}

#include <string>

namespace Sapphire::Graphics::Shaders {
	inline const std::string particleVertexModifierSource = "void ModifyVertex(thread float3& position, thread float3& normal, thread float4& color, float4 instanceData, float time, float4x4 model) { color *= instanceData; }";

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
		void ModifyVertex(thread float3& position, thread float3& normal, thread float4& color, float4 instanceData, float time, float4x4 model) {}
		#endif
	)";

	inline const std::string meshVertexPreparationSource = R"(
		struct VertexInput { float3 position [[attribute(0)]]; float3 normal [[attribute(1)]]; float2 uv [[attribute(2)]]; float4 color [[attribute(3)]];
		#if SKINNED
		ushort4 boneIndices [[attribute(4)]]; float4 boneWeights [[attribute(5)]];
		#endif
		#if INSTANCED
		float3 instancePosition [[attribute(4)]]; float3 instanceRotation [[attribute(5)]]; float3 instanceScale [[attribute(6)]]; float4 instanceData [[attribute(7)]];
		#endif
		};
		void PrepareVertex(VertexInput input, float4 uvTransform, float4 displacementParameters, float4x4 model,
		#if SKINNED
		device const float4x4* boneMatrices,
		#endif
		texture2d<float> heightMap, sampler heightSampler, texture2d<float> maskMap, sampler maskSampler,
		thread float3& position, thread float3& normal, thread float2& uv, thread float4& color, thread float4& instanceData) {
			uv = input.uv * uvTransform.xy + uvTransform.zw;
			float3 localPosition = input.position;
			normal = input.normal;
			color = input.color;
			#if SKINNED
			float4 skinnedPosition = float4(0.0);
			float3 skinnedNormal = float3(0.0);
			for (uint influence = 0; influence < 4; ++influence) {
				float weight = input.boneWeights[influence];
				skinnedPosition += (boneMatrices[input.boneIndices[influence]] * float4(localPosition, 1.0)) * weight;
				skinnedNormal += (boneMatrices[input.boneIndices[influence]] * float4(normal, 0.0)).xyz * weight;
			}
			localPosition = skinnedPosition.xyz;
			normal = normalize(skinnedNormal);
			#endif
			#if INSTANCED
			localPosition = input.instancePosition + InstanceRotate(localPosition * input.instanceScale, input.instanceRotation);
			normal = InstanceNormal(normal, input.instanceScale, input.instanceRotation);
			instanceData = input.instanceData;
			#else
			instanceData = float4(0.0);
			#endif
			position = Deform(localPosition, normal, uv, color, model, displacementParameters, heightMap, heightSampler, maskMap, maskSampler);
		}
	)";
	
	inline const std::string vertexShaderSource = deformationShaderSource + meshVertexPreparationSource + R"(
		struct VertexOutput { float4 position [[position]]; float2 uv; float4 color; float3 worldPosition; float3 worldNormal; float3 shadowPosition;
		#if CUSTOM_FRAGMENT_NORMAL
		float3 baseWorldPosition;
		#endif
		#ifdef SAMPLED_MODIFIER
		float4 modifierData;
		#endif
		#if SINGLE_LAYER
		uint layerInstance [[flat]];
		#endif
		};
		struct ObjectUniforms { float4x4 mvp; float4x4 model; float4x4 normal; float4 displacementParameters; float4 uvTransform; float4 modifierTime; uint4 layerParameters; };
		vertex VertexOutput main0(VertexInput input [[stage_in]], constant ObjectUniforms& uniforms [[buffer(0)]], constant float4* shadowBasis [[buffer(1)]],
		#if SKINNED
		device const float4x4* boneMatrices [[buffer(2)]],
		#endif
		#if SINGLE_LAYER
		uint instanceID [[instance_id]],
		#endif
		texture2d<float> heightMap [[texture(0)]], sampler heightSampler [[sampler(0)]], texture2d<float> maskMap [[texture(1)]], sampler maskSampler [[sampler(1)]]) {
			VertexOutput output;
			float2 uv;
			float3 localNormal;
			float4 color;
			float3 position;
			float4 instanceData;
			PrepareVertex(input, uniforms.uvTransform, uniforms.displacementParameters, uniforms.model,
			#if SKINNED
				boneMatrices,
			#endif
				heightMap, heightSampler, maskMap, maskSampler, position, localNormal, uv, color, instanceData);
			#if CUSTOM_FRAGMENT_NORMAL
			output.baseWorldPosition = (uniforms.model * float4(position, 1.0)).xyz;
			#endif
			#ifdef SAMPLED_MODIFIER
			ModifySampledVertex(position, uniforms.modifierTime.x, uniforms.model, heightMap, output.modifierData);
			#else
			ModifyVertex(position, localNormal, color, instanceData, uniforms.modifierTime.x, uniforms.model);
			#endif
			output.position = uniforms.mvp * float4(position, 1.0);
			#if SINGLE_LAYER
			output.layerInstance = uniforms.layerParameters.x + instanceID;
			#endif
			output.uv = uv;
			output.color = color;
			output.worldPosition = (uniforms.model * float4(position, 1.0)).xyz;
			output.worldNormal = normalize((uniforms.normal * float4(localNormal, 0.0)).xyz);
			output.shadowPosition = float3(dot(output.worldPosition, shadowBasis[0].xyz), dot(output.worldPosition, shadowBasis[1].xyz), dot(output.worldPosition, shadowBasis[2].xyz));
			return output;
		}
	)";

	inline const std::string fogCommonShaderSource = R"(
		#include <metal_stdlib>
		using namespace metal;
		struct FrameUniforms {
			float4 cameraPosition; float4 modifierTime; float4 fogColor; float4 fogParameters; float4 fogHeightParameters;
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
		float3 EncodeColor(float3 color, constant FrameUniforms& frame, float2 pixelPosition) {
			color = max(color, float3(0.0));
			if (frame.fogHeightParameters.w < 0.5) color = select(1.055 * pow(color, float3(1.0 / 2.4)) - 0.055, color * 12.92, color <= 0.0031308);
			uint2 pixel = uint2(pixelPosition);
			uint noiseBits = pixel.x * 0x8da6b343u ^ pixel.y * 0xd8163841u;
			noiseBits ^= noiseBits >> 16;
			noiseBits *= 0x7feb352du;
			noiseBits ^= noiseBits >> 15;
			float dither = (float(noiseBits & 0x00ffffffu) / 16777216.0 - 0.5) / 510.0;
			return max(color + dither, float3(0.0));
		}
	)";

	inline const std::string fogComputeShaderSource = fogCommonShaderSource + R"(
		struct FogDensityInputUniforms { float4 boundsMinimum; float4 boundsMaximum; float4 parameters; float4 grid; };
		struct FogComputeUniforms { FrameUniforms frame; uint4 densityInputCount; FogDensityInputUniforms densityInputs[5]; uint4 sliceMasks[64]; };
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
			float3 baseLighting = frame.ambientLight.rgb * frame.ambientLight.a;
			float directionalIntensity = frame.directionalDirection.w;
			if (directionalIntensity > 0.0) baseLighting += frame.directionalColor.rgb * directionalIntensity;
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
				float3 lighting = baseLighting;
				uint sliceMask = uniforms.sliceMasks[slice / 4][slice % 4];
				while (sliceMask != 0u) {
					uint light = sliceMask == 1u ? 0u : ctz(sliceMask);
					sliceMask &= sliceMask - 1u;
					float4 positionRange = frame.pointLightPositionRange[light];
					float4 colorIntensity = frame.pointLightColorIntensity[light];
					if (positionRange.w <= 0.0 || colorIntensity.w <= 0.0) continue;
					float3 toLight = positionRange.xyz - position;
					float distanceSquared = dot(toLight, toLight);
					if (distanceSquared >= positionRange.w * positionRange.w) continue;
					float rangeFade = clamp(1.0 - sqrt(distanceSquared) / positionRange.w, 0.0, 1.0);
					float coreRadius = max(2.0, positionRange.w * 0.25);
					float attenuation = rangeFade * rangeFade / (distanceSquared + coreRadius * coreRadius);
					lighting += colorIntensity.rgb * colorIntensity.w * attenuation;
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
			color = EncodeColor(color, frame, position.xy);
			return float4(color, 1.0);
		}
	)";

	inline const std::string fragmentShaderSource = fogCommonShaderSource + R"(
		struct FragmentInput { float4 position [[position]]; float2 uv; float4 color; float3 worldPosition; float3 worldNormal; float3 shadowPosition;
		#if CUSTOM_FRAGMENT_NORMAL
		float3 baseWorldPosition;
		#endif
		#ifdef SAMPLED_MODIFIER
		float4 modifierData;
		#endif
		#ifdef SINGLE_LAYER_VISIBILITY_SLOT
		uint layerInstance [[flat]];
		#endif
		};
		struct FragmentUniforms { float4 shadowParameters[3]; float4 shadowScale[3]; float4 shadowOffset[3]; packed_float3 lightingParameters; uint pointLightMask; float4 baseColor; float4 surfaceParameters; float4 customData[4]; };
		#ifndef CUSTOM_FRAGMENT_NORMAL
		void ModifyFragmentNormal(thread float3& normal, float3 baseWorldPosition, float3 worldPosition, float2 uv, float time) {}
		#endif
		float ShadowProjection(float3 position, float4 scale, float4 offset, float4 parameters, thread float2& uv, thread float& depth) {
			if (parameters.x < 0.5) return 0.0;
			float3 projected = position * scale.xyz + offset.xyz;
			uv = projected.xy;
			depth = projected.z;
			float border = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
			if (projected.z < 0.0 || (projected.z > 1.0 && parameters.w < 0.5)) return 0.0;
			return smoothstep(parameters.z * 2.0, parameters.z * 6.0, border);
		}
		float ShadowTaps(float depth, float2 uv, depth2d<float> shadowMap, sampler shadowSampler, float4 parameters) {
			float2 texelPosition = uv / parameters.z + 0.5;
			float2 base = floor(texelPosition);
			float2 fraction = texelPosition - base;
			float2 weight0 = 3.0 - 2.0 * fraction;
			float2 weight1 = 1.0 + 2.0 * fraction;
			float2 offset0 = (2.0 - fraction) / weight0 - 1.0;
			float2 offset1 = fraction / weight1 + 1.0;
			float2 baseUV = (base - 0.5) * parameters.z;
			float compareDepth = min(depth, 1.0) - parameters.y;
			float visibility = 0.0;
			visibility += weight0.x * weight0.y * shadowMap.sample_compare(shadowSampler, baseUV + float2(offset0.x, offset0.y) * parameters.z, compareDepth);
			visibility += weight1.x * weight0.y * shadowMap.sample_compare(shadowSampler, baseUV + float2(offset1.x, offset0.y) * parameters.z, compareDepth);
			visibility += weight0.x * weight1.y * shadowMap.sample_compare(shadowSampler, baseUV + float2(offset0.x, offset1.y) * parameters.z, compareDepth);
			visibility += weight1.x * weight1.y * shadowMap.sample_compare(shadowSampler, baseUV + float2(offset1.x, offset1.y) * parameters.z, compareDepth);
			return 1.0 - visibility * 0.0625;
		}
		float ShadowAmount(float3 position, float4 scale, float4 offset, depth2d<float> shadowMap, sampler shadowSampler, float4 parameters) {
			float2 uv;
			float depth;
			if (ShadowProjection(position, scale, offset, parameters, uv, depth) == 0.0) return 0.0;
			return ShadowTaps(depth, uv, shadowMap, shadowSampler, parameters);
		}
		float PointShadowVisibility(float3 direction, float receiverDistance, float range, float bias, float4 shadowInfo, depth2d_array<float> shadowMap, sampler shadowSampler) {
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
			float staticVisibility = (uint(shadowInfo.z) & (1u << face)) != 0u ? shadowMap.sample_compare(shadowSampler, uv * 0.5 + 0.5, uint(shadowInfo.x) * 6u + face, compareDepth) : 1.0;
			float dynamicVisibility = (uint(shadowInfo.w) & (1u << face)) != 0u ? shadowMap.sample_compare(shadowSampler, uv * 0.5 + 0.5, uint(shadowInfo.y) * 6u + face, compareDepth) : 1.0;
			return min(staticVisibility, dynamicVisibility);
		}
		float PointShadowAmount(float3 direction, float receiverDistance, float range, float bias, float4 shadowInfo, depth2d_array<float> shadowMap, sampler shadowSampler, float texelSize) {
			if (receiverDistance <= 0.0001 || receiverDistance >= range) return 0.0;
			float3 normalizedDirection = direction / receiverDistance;
			float3 axis = abs(normalizedDirection.y) < 0.99 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
			float3 tangent = normalize(cross(normalizedDirection, axis));
			float3 bitangent = cross(normalizedDirection, tangent);
			float filterSize = texelSize * 2.0;
			float visibility = 0.0;
			visibility += PointShadowVisibility(normalize(normalizedDirection + (tangent + bitangent) * filterSize), receiverDistance, range, bias, shadowInfo, shadowMap, shadowSampler);
			visibility += PointShadowVisibility(normalize(normalizedDirection + (tangent - bitangent) * filterSize), receiverDistance, range, bias, shadowInfo, shadowMap, shadowSampler);
			visibility += PointShadowVisibility(normalize(normalizedDirection + (-tangent + bitangent) * filterSize), receiverDistance, range, bias, shadowInfo, shadowMap, shadowSampler);
			visibility += PointShadowVisibility(normalize(normalizedDirection - (tangent + bitangent) * filterSize), receiverDistance, range, bias, shadowInfo, shadowMap, shadowSampler);
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
		float3 EvaluateLight(Surface surface, float3 light, bool twoSided, float nDotL) {
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


		fragment float4 main0(FragmentInput input [[stage_in]], constant FragmentUniforms& uniforms [[buffer(0)]], constant FrameUniforms& frame [[buffer(1)]], texture2d<float> image [[texture(0)]], sampler imageSampler [[sampler(0)]], depth2d<float> detailMap [[texture(1)]], sampler detailSampler [[sampler(1)]], depth2d<float> fallbackMap [[texture(2)]], sampler fallbackSampler [[sampler(2)]], depth2d<float> dynamicMap [[texture(3)]], sampler dynamicSampler [[sampler(3)]], depth2d_array<float> pointShadowMap [[texture(4)]], sampler pointShadowSampler [[sampler(4)]], texture2d<float> normalMap [[texture(5)]], sampler normalSampler [[sampler(5)]], texture2d<float> roughnessMap [[texture(6)]], sampler roughnessSampler [[sampler(6)]], texture2d<float> aoMap [[texture(7)]], sampler aoSampler [[sampler(7)]], texture3d<float> fogVolumeMap [[texture(8)]], sampler fogVolumeSampler [[sampler(8)]]
		#if CUSTOM_DEPTH_SURFACE
			, depth2d<float> sceneDepth [[texture(9)]]
		#endif
		#ifdef SINGLE_LAYER_VISIBILITY_SLOT
			, texture2d<uint, access::read> layerVisibility [[texture(SINGLE_LAYER_VISIBILITY_SLOT)]], uint primitiveID [[primitive_id]]
		#endif
		) {
			#ifdef SINGLE_LAYER_VISIBILITY_SLOT
			uint2 nearest = layerVisibility.read(uint2(input.position.xy)).xy;
			bool behindNearest = nearest.x != 0u && any(nearest != uint2(input.layerInstance, primitiveID));
			#endif
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
			#if CUSTOM_FRAGMENT_NORMAL
			if ((materialFlags & 16u) == 0u) ModifyFragmentNormal(normal, input.baseWorldPosition, input.worldPosition, input.uv, frame.modifierTime.x);
			normal = normalize(normal);
			#endif
			float4 texel = image.sample(imageSampler, input.uv);
			#if ALPHA_MASKED
			if (texel.a < 0.5) discard_fragment();
			#endif
			float fog = 0.0;
			float3 fogScattering = float3(0.0);
			float fogTransmittance = 1.0;
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
			else {
				fog = HeightFog(input.worldPosition, frame);
				fogTransmittance = 1.0 - fog;
			}
			float opacity = input.color.a * texel.a * uniforms.lightingParameters.y;
			#ifdef SINGLE_LAYER_VISIBILITY_SLOT
			if (behindNearest && ((materialFlags & 16u) != 0u || (fog >= 1.0 && frame.fogVolumeParameters.x < 0.5))) discard_fragment();
			#endif
			if (fog >= 1.0 && frame.fogVolumeParameters.x < 0.5) return float4(EncodeColor(frame.fogColor.rgb, frame, input.position.xy), opacity);
			if ((materialFlags & 16u) != 0u) {
				float3 tint = input.color.rgb * uniforms.baseColor.rgb * texel.rgb;
				float3 color = frame.fogVolumeParameters.x > 0.5 ? tint * fogTransmittance + fogScattering : mix(tint, frame.fogColor.rgb, fog);
				return float4(EncodeColor(color, frame, input.position.xy), opacity);
			}
			float3 albedo = input.color.rgb * texel.rgb * uniforms.baseColor.rgb;
			float roughness = clamp(uniforms.surfaceParameters.x * ((materialFlags & 2u) != 0u ? roughnessMap.sample(roughnessSampler, input.uv).r : 1.0), 0.045, 1.0);
			float3 view = frame.cameraPosition.xyz - input.worldPosition;
			view *= rsqrt(max(dot(view, view), 0.000001));
			float3 surfaceRadiance = float3(0.0);
			#if CUSTOM_DEPTH_SURFACE
			constexpr sampler depthSampler(coord::normalized, address::clamp_to_edge, filter::nearest);
			float depth = sceneDepth.sample(depthSampler, input.position.xy / frame.fogScreenParameters.xy, level(0.0));
			float opaqueDistance = -1.0;
			if (depth < 1.0 && frame.cameraForward.w > 0.5) {
				float nearClip = frame.fogViewParameters.z;
				float farClip = frame.fogViewParameters.w;
				float range = farClip - nearClip;
				float opaqueViewDepth = nearClip * farClip / (farClip - depth * range);
				float surfaceViewDepth = nearClip * farClip / (farClip - input.position.z * range);
				opaqueDistance = max(opaqueViewDepth - surfaceViewDepth, 0.0) / max(dot(-view, frame.cameraForward.xyz), 0.0001);
			}
			ModifyDepthSurface(albedo, normal, roughness, surfaceRadiance, input.worldPosition, view, opaqueDistance, uniforms.customData, frame
			#ifdef SAMPLED_MODIFIER
				, normalMap, input.modifierData
			#endif
			);
			normal = normalize(normal);
			roughness = clamp(roughness, 0.045, 1.0);
			#endif
			#ifdef SINGLE_LAYER_VISIBILITY_SLOT
			if (behindNearest) discard_fragment();
			#endif
			float nDotV = max(twoSided ? abs(dot(normal, view)) : dot(normal, view), 0.0001);
			float ao = (materialFlags & 4u) != 0u ? mix(1.0, aoMap.sample(aoSampler, input.uv).r, uniforms.surfaceParameters.z) : 1.0;
			float metallic = uniforms.baseColor.w;
			float alphaSquared = roughness * roughness * roughness * roughness;
			Surface surface{normal, view, albedo, mix(float3(0.04), albedo, metallic), metallic, alphaSquared, nDotV, sqrt(nDotV * nDotV * (1.0 - alphaSquared) + alphaSquared)};
			float diffuse = max(twoSided ? abs(dot(normal, -frame.directionalDirection.xyz)) : dot(normal, -frame.directionalDirection.xyz), 0.0);
			bool directionalShadow = uniforms.lightingParameters.z > 0.5 && diffuse * frame.directionalDirection.w > 0.0;
			float staticShadow = 0.0;
			if (directionalShadow) {
				float2 detailUV;
				float detailDepth;
				float detailCoverage = ShadowProjection(input.shadowPosition, uniforms.shadowScale[0], uniforms.shadowOffset[0], uniforms.shadowParameters[0], detailUV, detailDepth);
				if (detailCoverage < 1.0) staticShadow = ShadowAmount(input.shadowPosition, uniforms.shadowScale[1], uniforms.shadowOffset[1], fallbackMap, fallbackSampler, uniforms.shadowParameters[1]);
				if (detailCoverage > 0.0) staticShadow = mix(staticShadow, ShadowTaps(detailDepth, detailUV, detailMap, detailSampler, uniforms.shadowParameters[0]), detailCoverage);
			}
			float dynamicShadow = 0.0;
			if (directionalShadow) dynamicShadow = ShadowAmount(input.shadowPosition, uniforms.shadowScale[2], uniforms.shadowOffset[2], dynamicMap, dynamicSampler, uniforms.shadowParameters[2]);
			float3 direct = frame.directionalDirection.w > 0.0 ? EvaluateLight(surface, -frame.directionalDirection.xyz, twoSided, diffuse) * frame.directionalColor.rgb * frame.directionalDirection.w * (1.0 - staticShadow) * (1.0 - dynamicShadow) : float3(0.0);
			float3 pointLighting = float3(0.0);
			uint lightMask = uniforms.pointLightMask;
			while (lightMask != 0u) {
				uint index = lightMask == 1u ? 0u : ctz(lightMask);
				lightMask &= lightMask - 1u;
				float4 positionRange = frame.pointLightPositionRange[index];
				float4 colorIntensity = frame.pointLightColorIntensity[index];
				float3 toLight = positionRange.xyz - input.worldPosition;
				float distanceSquared = dot(toLight, toLight);
				if (distanceSquared < positionRange.w * positionRange.w) {
					float distanceToLight = sqrt(distanceSquared);
					float3 lightDirection = toLight / max(distanceToLight, 0.0001);
					float pointDiffuse = max(twoSided ? abs(dot(normal, lightDirection)) : dot(normal, lightDirection), 0.0);
					if (pointDiffuse <= 0.0) continue;
					float rangeFade = clamp(1.0 - distanceToLight / positionRange.w, 0.0, 1.0);
					float attenuation = rangeFade * rangeFade / max(distanceSquared, 1.0);
					float4 shadowInfo = frame.pointLightShadowInfo[index];
					float shadowBias = frame.pointLightShadowBias[index].x;
					bool staticShadowValid = uniforms.lightingParameters.z > 0.5 && shadowInfo.z > 0.5;
					bool dynamicShadowValid = uniforms.lightingParameters.z > 0.5 && shadowInfo.w > 0.5;
					float receiverDistance = staticShadowValid || dynamicShadowValid ? distanceToLight : 0.0;
					float pointShadow = staticShadowValid || dynamicShadowValid ? PointShadowAmount(-toLight, receiverDistance, positionRange.w, shadowBias, shadowInfo, pointShadowMap, pointShadowSampler, uniforms.shadowParameters[0].z) : 0.0;
					pointLighting += EvaluateLight(surface, lightDirection, twoSided, pointDiffuse) * colorIntensity.rgb * colorIntensity.a * attenuation * (1.0 - pointShadow);
				}
			}
			float3 ambient = frame.ambientLight.a > 0.5 ? frame.ambientLight.rgb : float3(0.0);
			float3 color = albedo * ambient * ao * (1.0 - metallic) + direct + pointLighting + surfaceRadiance;

			color = frame.fogVolumeParameters.x > 0.5 ? color * fogTransmittance + fogScattering : mix(color, frame.fogColor.rgb, fog);
			return float4(EncodeColor(color, frame, input.position.xy), opacity);
		}
	)";

	inline const std::string shadowVertexShaderSource = deformationShaderSource + meshVertexPreparationSource + R"(
		struct VertexOutput { float4 position [[position]]; float2 uv; };
		struct ShadowUniforms { float4x4 mvp; float4x4 model; float4 displacementParameters; float4 uvTransform; float4 modifierTime; };
		vertex VertexOutput main0(VertexInput input [[stage_in]], constant ShadowUniforms& uniforms [[buffer(0)]],
		#if SKINNED
		device const float4x4* boneMatrices [[buffer(1)]],
		#endif
		texture2d<float> heightMap [[texture(0)]], sampler heightSampler [[sampler(0)]], texture2d<float> maskMap [[texture(1)]], sampler maskSampler [[sampler(1)]]) {
			VertexOutput output;
			float2 uv;
			float3 localNormal;
			float4 color;
			float3 position;
			float4 instanceData;
			PrepareVertex(input, uniforms.uvTransform, uniforms.displacementParameters, uniforms.model,
			#if SKINNED
				boneMatrices,
			#endif
				heightMap, heightSampler, maskMap, maskSampler, position, localNormal, uv, color, instanceData);
			#ifdef SAMPLED_MODIFIER
			float4 modifierData;
			ModifySampledVertex(position, uniforms.modifierTime.x, uniforms.model, heightMap, modifierData);
			#else
			ModifyVertex(position, localNormal, color, instanceData, uniforms.modifierTime.x, uniforms.model);
			#endif
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

	inline constexpr const char* layerDepthFragmentShaderSource = R"(
		#include <metal_stdlib>
		using namespace metal;
		struct FragmentInput { float4 position [[position]]; float2 uv; float4 color; uint layerInstance [[flat]]; };
		fragment uint2 main0(FragmentInput input [[stage_in]], constant float4& opacity [[buffer(0)]], texture2d<float> image [[texture(0)]], sampler imageSampler [[sampler(0)]], uint primitiveID [[primitive_id]]) {
			if (input.color.a * image.sample(imageSampler, input.uv).a * opacity.x <= 0.0) discard_fragment();
			return uint2(input.layerInstance, primitiveID);
		}
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

	inline const std::string waterSpectrumSource = R"(
		#define SAMPLED_MODIFIER 1
		float3 OceanSample(texture2d<float> atlas, float4 data, int plane, float lod) {
			constexpr sampler linearSampler(coord::normalized, filter::linear, mip_filter::linear);
			uint first = uint(floor(data.z)) % 64u;
			uint next = (first + 1u) % 64u;
			float2 inside = fract(data.xy) * 128.0 + 1.5;
			float2 a = float2(first % 8u + uint(plane * 8), first / 8u);
			float2 b = float2(next % 8u + uint(plane * 8), next / 8u);
			float2 dimensions = float2(atlas.get_width(), atlas.get_height());
			float3 left = atlas.sample(linearSampler, (130.0 * a + inside) / dimensions, level(lod)).rgb;
			float3 right = atlas.sample(linearSampler, (130.0 * b + inside) / dimensions, level(lod)).rgb;
			return mix(left, right, fract(data.z)) * 2.0 - 1.0;
		}
	)";

	inline const std::string waterVertexModifierSource = waterSpectrumSource + R"(
		void ModifySampledVertex(thread float3& position, float time, float4x4 model, texture2d<float> atlas, thread float4& data) {
			float2 wind = float2(WIND_X, WIND_Y);
			float2 across = float2(-wind.y, wind.x);
			float2 world = (model * float4(position, 1.0)).xy;
			float height = length(model[2].xyz) * 0.5;
			data = float4(float2(dot(world, wind), dot(world, across)) / PATCH_METRES,
				fract(time * sqrt(160.0 / PATCH_METRES) / 16.0) * 64.0, height / PATCH_METRES);
			position.z += 0.5 * OceanSample(atlas, data, 0, 0.0).r;
			float2 horizontal = height * CHOP * (wind * OceanSample(atlas, data, 1, 0.0).r + across * OceanSample(atlas, data, 2, 0.0).r);
			position.x += dot(horizontal, model[0].xy) / max(dot(model[0].xyz, model[0].xyz), 1e-12);
			position.y += dot(horizontal, model[1].xy) / max(dot(model[1].xyz, model[1].xyz), 1e-12);
		}
	)";

	inline const std::string waterSurfaceSource = waterSpectrumSource + R"(
		void ModifyDepthSurface(thread float3& albedo, thread float3& normal, thread float& roughness, thread float3& radiance,
			float3 position, float3 view, float opaqueDistance, constant float4* parameters, constant FrameUniforms& frame,
			texture2d<float> atlas, float4 data) {
			float4 shoreline = parameters[0];
			float footprint = max(length(dfdx(data.xy)), length(dfdy(data.xy))) * 128.0;
			float lod = clamp(log2(max(footprint, 1.0)), 0.0, 1.0);
			float scale = 64.0 * data.w;
			float2 dz = OceanSample(atlas, data, 0, lod).gb * scale;
			float2 dx = OceanSample(atlas, data, 1, lod).gb * scale * parameters[1].z;
			float2 dy = OceanSample(atlas, data, 2, lod).gb * scale * parameters[1].z;
			float3 waveNormal = cross(float3(1.0 + dx.x, dy.x, dz.x), float3(dx.y, 1.0 + dy.y, dz.y));
			float2 wind = parameters[1].xy;
			float2 xy = wind * waveNormal.x + float2(-wind.y, wind.x) * waveNormal.y;
			float detail = 1.0 - smoothstep(2.0, 10.0, footprint);
			normal = normalize(float3(xy * detail, max(waveNormal.z, 0.01)));
			float rippleTime = frame.modifierTime.x;
			float depth = opaqueDistance < 0.0 ? shoreline.y : opaqueDistance * max(abs(view.z), 0.25);
			float shallows = 1.0 - smoothstep(0.0, shoreline.y, depth);
			float3 waterColor = albedo;
			albedo = mix(albedo, sqrt(saturate(waterColor)), shallows * 0.58);

			float contactFilter = fwidth(max(opaqueDistance, 0.0));
			float foam = 0.0;
			if (shoreline.z > 0.5 && shoreline.x > 0.0 && opaqueDistance >= 0.0) {
				float pulse = sin(dot(position.xy, float2(3.2, 2.1)) - rippleTime * 4.0);
				float breakup = sin(dot(position.xy, float2(-4.1, 3.3)) + rippleTime * 3.1);
				float width = shoreline.x * (0.75 + 0.16 * pulse + 0.09 * breakup);
				float contact = 1.0 - smoothstep(width * 0.35, width + min(contactFilter, width * 0.25), opaqueDistance);
				foam = contact * (0.53 + 0.19 * pulse + 0.17 * breakup);
			}
			albedo = mix(albedo, mix(waterColor, float3(1.0), 0.48), foam);
			roughness = mix(roughness, 0.65, foam);
			float grazing = 1.0 - saturate(dot(normal, view));
			float fresnel = 0.02 + 0.98 * pow(grazing, 5.0);
			float3 ambient = frame.ambientLight.rgb * frame.ambientLight.a;
			float3 daylight = frame.directionalColor.rgb * frame.directionalDirection.w * saturate(-frame.directionalDirection.z);
			float3 scatter = waterColor * (ambient + daylight) * (0.35 + 0.45 * shallows);
			float horizon = 1.0 - saturate(reflect(-view, normal).z);
			radiance = mix(scatter, ambient * (0.4 + 0.6 * horizon), fresnel) * (1.0 - foam);
		}
	)";
}
