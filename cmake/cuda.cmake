


function(find_cuda)
    if(NOT BPFTIME_CUDA_ROOT)
        message(FATAL_ERROR "To use NV attach, set BPFTIME_CUDA_ROOT to the root of CUDA installation, such as /usr/local/cuda-12.6")
    endif()

    # Detect target platform based on CMAKE_SYSTEM_PROCESSOR
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(CUDA_TARGET_ARCH "aarch64-linux")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
        set(CUDA_TARGET_ARCH "x86_64-linux")
    else()
        message(WARNING "Unsupported architecture ${CMAKE_SYSTEM_PROCESSOR}, defaulting to x86_64-linux")
        set(CUDA_TARGET_ARCH "x86_64-linux")
    endif()

    # Library layout differs between installer types:
    # - runfile installs usually use /usr/local/cuda-*/lib64 + extras/CUPTI/lib64
    # - deb installs use /usr/local/cuda-*/targets/<arch>/lib
    set(_cuda_library_paths
        ${BPFTIME_CUDA_ROOT}/lib64
        ${BPFTIME_CUDA_ROOT}/targets/${CUDA_TARGET_ARCH}/lib
        ${BPFTIME_CUDA_ROOT}/targets/${CUDA_TARGET_ARCH}/lib/stubs
        ${BPFTIME_CUDA_ROOT}/extras/CUPTI/lib64
    )
    set(CUDA_LIBRARY_PATH ${_cuda_library_paths} PARENT_SCOPE)

    # Detect CUDA version from version.json or version.txt
    if(EXISTS "${BPFTIME_CUDA_ROOT}/version.json")
        file(READ "${BPFTIME_CUDA_ROOT}/version.json" CUDA_VERSION_JSON)
        # Match the cuda section and extract version number (e.g., "12.8.1" -> major=12, minor=8)
        string(REGEX MATCH "\"cuda\"[^{]*\\{[^}]*\"version\"[^\"]*\"([0-9]+)\\.([0-9]+)" _ "${CUDA_VERSION_JSON}")
        set(CUDA_VERSION_MAJOR ${CMAKE_MATCH_1})
        set(CUDA_VERSION_MINOR ${CMAKE_MATCH_2})
    elseif(EXISTS "${BPFTIME_CUDA_ROOT}/version.txt")
        file(READ "${BPFTIME_CUDA_ROOT}/version.txt" CUDA_VERSION_TXT)
        string(REGEX MATCH "CUDA Version ([0-9]+)\\.([0-9]+)" _ "${CUDA_VERSION_TXT}")
        set(CUDA_VERSION_MAJOR ${CMAKE_MATCH_1})
        set(CUDA_VERSION_MINOR ${CMAKE_MATCH_2})
    else()
        # Try to extract from path name as fallback (e.g., cuda-13.0)
        string(REGEX MATCH "cuda-([0-9]+)\\.([0-9]+)" _ "${BPFTIME_CUDA_ROOT}")
        set(CUDA_VERSION_MAJOR ${CMAKE_MATCH_1})
        set(CUDA_VERSION_MINOR ${CMAKE_MATCH_2})
    endif()

    # Header layout differs between installer types:
    # - runfile installs usually use ${BPFTIME_CUDA_ROOT}/include + extras/CUPTI/include
    # - deb installs use ${BPFTIME_CUDA_ROOT}/targets/<arch>/include (contains CUDA + CUPTI headers)
    set(_cuda_include_paths
        ${BPFTIME_CUDA_ROOT}/include
        ${BPFTIME_CUDA_ROOT}/targets/${CUDA_TARGET_ARCH}/include
        ${BPFTIME_CUDA_ROOT}/extras/CUPTI/include
    )
    set(CUDA_INCLUDE_PATH ${_cuda_include_paths} PARENT_SCOPE)

    message(STATUS "Detected CUDA version: ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}")

    # Resolve CUPTI library (prefer static when present).
    find_library(_cupti_static_lib
        NAMES cupti_static
        HINTS ${_cuda_library_paths}
        NO_DEFAULT_PATH
    )
    find_library(_cupti_shared_lib
        NAMES cupti
        HINTS ${_cuda_library_paths}
        NO_DEFAULT_PATH
    )
    if(_cupti_static_lib)
        set(_cuda_cupti_lib "${_cupti_static_lib}")
        message(STATUS "CUDA ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}: Using CUPTI static: ${_cuda_cupti_lib}")
    elseif(_cupti_shared_lib)
        set(_cuda_cupti_lib "${_cupti_shared_lib}")
        message(STATUS "CUDA ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}: Using CUPTI shared: ${_cuda_cupti_lib}")
    else()
        message(FATAL_ERROR "Could not find CUPTI library (libcupti.so or libcupti_static.a). Check BPFTIME_CUDA_ROOT=${BPFTIME_CUDA_ROOT}.")
    endif()

    # Resolve nvPTXCompiler library. CUDA 12.x typically provides libnvptxcompiler_static.a.
    find_library(_nvptxcompiler_static_lib
        NAMES nvptxcompiler_static
        HINTS ${_cuda_library_paths}
        NO_DEFAULT_PATH
    )
    find_library(_nvptxcompiler_shared_lib
        NAMES nvptxcompiler
        HINTS ${_cuda_library_paths}
        NO_DEFAULT_PATH
    )
    if(_nvptxcompiler_static_lib)
        set(_cuda_nvptxcompiler_lib "${_nvptxcompiler_static_lib}")
        message(STATUS "CUDA ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}: Using nvptxcompiler static: ${_cuda_nvptxcompiler_lib}")
    elseif(_nvptxcompiler_shared_lib)
        set(_cuda_nvptxcompiler_lib "${_nvptxcompiler_shared_lib}")
        message(STATUS "CUDA ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}: Using nvptxcompiler shared: ${_cuda_nvptxcompiler_lib}")
    else()
        message(FATAL_ERROR "Could not find nvPTXCompiler library (libnvptxcompiler.so or libnvptxcompiler_static.a). Check BPFTIME_CUDA_ROOT=${BPFTIME_CUDA_ROOT}.")
    endif()

    # Keep CUDA driver/runtime and NVRTC as link items; use absolute paths for CUPTI/nvPTXCompiler.
    set(CUDA_LIBS cuda cudart ${_cuda_nvptxcompiler_lib} ${_cuda_cupti_lib} nvrtc PARENT_SCOPE)
endfunction()
