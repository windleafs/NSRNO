"""Use the identical capture for either existing 60-us or 90-us acquisition."""
from .native_capture import capture as base_capture


def capture(material,source,si,sw,ri,rw,config,frequencies,stride=10):
    fields,raw=base_capture(material,source,si,sw,ri,rw,config,frequencies,stride)
    samples=(config.nt-1)//stride+1
    fields*=2401/samples
    return fields,raw
