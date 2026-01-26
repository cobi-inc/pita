# TensorRT Licensing Notice

**Important information about using PITA's optional TensorRT backend**

## Overview

PITA's TensorRT backend is an **optional feature** that provides inference using NVIDIA's TensorRT-LLM library. While TensorRT-LLM itself is Apache 2.0 licensed, it depends on NVIDIA's proprietary **TensorRT** runtime, which has separate licensing requirements.

## TensorRT Proprietary License

### NVIDIA's License Terms

NVIDIA TensorRT is proprietary software licensed under NVIDIA's Software License Agreement (SLA). Key restrictions include:

- ❌ Cannot redistribute TensorRT
- ❌ Cannot create derivative works of TensorRT
- ❌ Cannot use in a way that "subjects it to an open source license requiring source redistribution"
- ❌ Not designed/tested for certain production business-critical systems (varies by version)
- ✅ Users must obtain TensorRT directly from NVIDIA and accept their terms

### Where to Find NVIDIA's License

- Official TensorRT page: https://developer.nvidia.com/tensorrt
- License agreement: https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/sla.html
- You must review and accept these terms when installing TensorRT

## Interaction with PITA's AGPLv3 License

### The Theoretical Question

NVIDIA's license states TensorRT cannot be "subjected to an open source license that requires the SOFTWARE to be redistributed in source code form." This raises a question: Is there a conflict when using AGPLv3-licensed PITA with proprietary TensorRT?

### Why This Is Likely Not a Problem

The overwhelming legal consensus is that this combination is permissible under the **"mere aggregation"** doctrine:

#### 1. Separate Software Components

TensorRT and PITA are distinct programs:
- PITA doesn't include TensorRT code
- Users install TensorRT separately from NVIDIA
- PITA doesn't redistribute TensorRT
- They run as separate processes

#### 2. AGPLv3 "Mere Aggregation" Exception

The AGPLv3 license explicitly allows "mere aggregation" of separate programs:

> "A compilation of a covered work with other separate and independent works... on a storage medium... is called an 'aggregate' if the compilation and its resulting copyright are not used to limit the access or legal rights of the compilation's users beyond what the individual works permit."

This means running PITA (AGPLv3) with TensorRT (proprietary) on the same system is generally considered acceptable.

#### 3. Dynamic Linking

PITA uses TensorRT via:
- Python imports (dynamic, at runtime)
- No static linking
- No compilation together
- Clean separation at runtime

This further supports the "separate programs" argument.

#### 4. Optional Dependency

TensorRT is **completely optional**:
- PITA works without TensorRT (vLLM, llama.cpp backends available)
- TensorRT support is a plugin-like feature
- Core PITA functionality doesn't require TensorRT

#### 5. Industry Precedent

This situation is common and generally accepted:

**Similar Situations Considered Legal:**
- ✅ AGPLv3 web applications using proprietary databases (Oracle, SQL Server, MongoDB Enterprise)
- ✅ AGPLv3 software using proprietary GPU drivers (NVIDIA CUDA drivers)
- ✅ AGPLv3 code running on proprietary operating systems (Windows, macOS)
- ✅ AGPLv3 applications calling proprietary APIs (cloud services, etc.)

All of these involve AGPLv3 code interacting with proprietary components via dynamic linking/runtime interfaces.

### Legal Opinion

Based on:
- The "mere aggregation" exception in AGPLv3
- Separation of components
- Industry practice
- Legal commentary on similar situations

**We believe this combination is legally sound.**

However, because:
- NVIDIA's terms are unusually explicit about open source licenses
- AG PLv3's network clause is stricter than standard GPL
- Limited specific case law on this exact scenario

**We recommend consulting legal counsel if you have concerns.**

### Comparison Chart

| Aspect | TensorRT + AGPLv3 PITA | Similar Precedent |
|--------|------------------------|-------------------|
| **Separate installation** | ✅ Yes | Oracle DB + AGPLv3 app |
| **No redistribution** | ✅ PITA doesn't distribute TensorRT | GPU drivers + Linux (GPL) |
| **Dynamic linking** | ✅ Python runtime imports | System libraries + GPL apps |
| **Optional component** | ✅ Can use other backends | Database choice in apps |
| **Mere aggregation** | ✅ Separate programs | Any proprietary OS + GPL software |

## Practical Guidance

### For AGPLv3 PITA Users

If you want to use the TensorRT backend with AGPLv3-licensed PITA:

1. **Install TensorRT separately from NVIDIA**
   - Don't expect PITA to bundle or provide TensorRT
   - Download from NVIDIA: https://developer.nvidia.com/tensorrt
   - Accept NVIDIA's license agreement

2. **Document your use**
   - Note that TensorRT is a separate component
   - Include both PITA's AGPLv3 notice and reference to NVIDIA's TensorRT license
   - Make clear they are separate programs

3. **If providing source to users (AGPLv3 requirement)**
   - Provide PITA source code (as required by AGPLv3)
   - Do NOT include TensorRT in your source distribution
   - Instruct users to obtain TensorRT from NVIDIA separately

4. **If you're concerned**
   - Consult with your legal counsel
   - Consider using vLLM or llama.cpp backends (both open source)
   - Consider obtaining PITA commercial license (removes AGPLv3 question)

### For Commercial PITA License Users

If you have a commercial PITA license:

1. **TensorRT licensing still separate**
   - You still need to obtain TensorRT from NVIDIA
   - NVIDIA's license terms still apply
   - Commercial PITA license doesn't include TensorRT rights

2. **No AGPLv3 interaction question**
   - Since you're not using AGPLv3 PITA license, no AGPLv3/TensorRT interaction
   - Only NVIDIA's license requirements apply (for TensorRT)

3. **Contact NVIDIA**
   - For commercial TensorRT use, review NVIDIA's terms
   - May need commercial TensorRT license depending on your use case

## Installation Requirements

### Technical Requirements

When using TensorRT with PITA:

```bash
# Install PITA without TensorRT first
pip install pita

# Install TensorRT separately from NVIDIA
# (Follow NVIDIA's installation instructions)
# Accept NVIDIA's license during installation

# Then install tensorrt_llm
pip install tensorrt_llm
```

**Note**: PITA's `pita[tensorrt]` option lists tensorrt_llm as a dependency, but users must still:
1. Install underlying TensorRT from NVIDIA separately
2. Accept NVIDIA's license agreement
3. Ensure they comply with both licenses

### Version Compatibility

Check PITA documentation for compatible TensorRT versions:
- Different versions may have different license terms
- Ensure compatibility with your TensorRT-LLM version

## Alternatives to TensorRT

If you're concerned about TensorRT licensing or compatibility:

### vLLM Backend (Recommended)
- **License**: Apache 2.0 (permissive, no conflicts)
- **Performance**: Excellent for GPU inference
- **Compatibility**: Works with AGPLv3 PITA seamlessly
- **Installation**: `pip install pita[vllm]`

### llama.cpp Backend
- **License**: MIT (permissive, no conflicts)
- **Performance**: Great for CPU inference, good for GPU
- **Compatibility**: Works with AGPLv3 PITA seamlessly
- **Installation**: `pip install pita[llama_cpp]`

Both alternatives are fully open source and have no licensing ambiguity.

## Summary

### Bottom Line

**Using TensorRT with AGPLv3 PITA is likely legally permissible** under the "mere aggregation" doctrine, similar to many common AGPLv3 + proprietary software combinations.

**However:**
- TensorRT must be obtained separately from NVIDIA
- Users must accept NVIDIA's license terms
- If concerned, consult legal counsel or use open source backends

### Decision Matrix

| Your Situation | Recommendation |
|----------------|----------------|
| **Research/academic use** | AGPLv3 PITA + TensorRT likely fine, document separation |
| **Commercial product (AGPLv3 compliant)** | Consult counsel if concerned, or use vLLM/llama.cpp |
| **Cannot disclose source** | Get commercial PITA license + obtain TensorRT from NVIDIA |
| **Want zero legal questions** | Use vLLM or llama.cpp backends (fully open source) |
| **Need commercial PITA** | Commercial license + separate TensorRT from NVIDIA |

## Additional Resources

### NVIDIA Resources
- TensorRT Home: https://developer.nvidia.com/tensorrt
- TensorRT License: https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/sla.html
- TensorRT-LLM (Apache 2.0): https://github.com/NVIDIA/TensorRT-LLM

### PITA Resources
- Complete licensing guide: [LICENSING-GUIDE.md](LICENSING-GUIDE.md)
- Main license: [LICENSE](LICENSE.md)
- Dependency notices: [NOTICE](NOTICE.md)
- Commercial licensing: sales@cobi-inc.com

### Legal Resources
- AGPLv3 full text: https://www.gnu.org/licenses/agpl-3.0.txt
- FSF GPL FAQ: https://www.gnu.org/licenses/gpl-faq.html
- "Mere aggregation" discussion: https://www.gnu.org/licenses/gpl-faq.html#MereAggregation

## Contact

For questions about TensorRT compatibility with PITA:

**Technical questions**: Open an issue at https://github.com/cobi-inc-MC/pita/issues
**Licensing questions**: sales@cobi-inc.com
**Legal concerns**: Consult with qualified legal counsel

---

**Disclaimer**: This document provides information about licensing considerations. It is not legal advice. For specific legal questions about your use case, consult with a qualified intellectual property attorney.

**Last Updated**: January 2026
