# Only used for PyTorch open source BUCK build

IGNORED_ATTRIBUTE_PREFIX = [
    "apple",
    "fbobjc",
    "windows",
    "fbandroid",
    "macosx",
]

IGNORED_ATTRIBUTES = [
    "feature",
    "platforms",
]

def filter_attributes(kwgs):
    keys = list(kwgs.keys())

    # drop unncessary attributes
    for key in keys:
        if key in IGNORED_ATTRIBUTES:
            kwgs.pop(key)
        else:
            for invalid_prefix in IGNORED_ATTRIBUTE_PREFIX:
                if key.startswith(invalid_prefix):
                    kwgs.pop(key)
    return kwgs

# map fbsource deps to OSS deps
def to_oss_deps(deps = []):
    new_deps = []
    for dep in deps:
        new_deps += map_deps(dep)
    return new_deps

def map_deps(dep):
    # keep relative root targets
    if dep.startswith(":"):
        return [dep]

    # remove @fbsource prefix
    if dep.startswith("@fbsource"):
        dep = dep[len("@fbsource"):]

    # ignore all fbsource linker_lib targets
    if dep.startswith("//xplat/third-party/linker_lib:"):
        return []

    # ignore all folly libraries
    if dep.startswith("//xplat/folly:"):
        return []

    fail("Unknown OSS BUCK dep " + dep)
    return []
