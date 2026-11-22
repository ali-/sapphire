cmake_minimum_required(VERSION 3.21)
project(Sapphire LANGUAGES NONE)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND NOT WIN32)
    message(FATAL_ERROR "Sapphire currently supports macOS and Windows.")
endif()

if(NOT TARGET SDL3::SDL3)
    find_package(SDL3 3.4.10 CONFIG REQUIRED)
endif()

add_library(sapphire INTERFACE)
add_library(Sapphire::Sapphire ALIAS sapphire)

target_include_directories(sapphire INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(sapphire INTERFACE cxx_std_20)
target_link_libraries(sapphire INTERFACE SDL3::SDL3)

if(WIN32)
    target_compile_definitions(sapphire INTERFACE NOMINMAX SAPPHIRE_SHADER_SOURCE="shaders_hlsl.cpp")
    target_link_libraries(sapphire INTERFACE dxcompiler)

    function(sapphire_copy_windows_runtime target)
        get_filename_component(sdk_root
            "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots;KitsRoot10]"
            ABSOLUTE)
        set(sdk_version "${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}")
        if(NOT sdk_version)
            set(sdk_version "$ENV{WindowsSDKVersion}")
            string(REGEX REPLACE "[/\\\\]+$" "" sdk_version "${sdk_version}")
        endif()
        string(TOLOWER "${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}" sdk_arch)
        set(dxc_dir "${sdk_root}/bin/${sdk_version}/${sdk_arch}")
        foreach(dll IN ITEMS dxcompiler.dll dxil.dll)
            if(NOT EXISTS "${dxc_dir}/${dll}")
                message(FATAL_ERROR "Could not find ${dll} in ${dxc_dir}. Select a Windows SDK with DXC.")
            endif()
            add_custom_command(TARGET "${target}" POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    "${dxc_dir}/${dll}" "$<TARGET_FILE_DIR:${target}>"
                VERBATIM)
        endforeach()
    endfunction()
else()
    target_compile_definitions(sapphire INTERFACE SAPPHIRE_SHADER_SOURCE="shaders_metal.cpp")
endif()
