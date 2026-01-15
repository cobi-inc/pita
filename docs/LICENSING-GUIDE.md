# PITA Licensing Guide

Complete guide to licensing for the PITA (Probabilistic Inference Time Algorithms) library.

## Table of Contents

- [Overview](#overview)
- [Dual Licensing Model](#dual-licensing-model)
- [Open Source License (AGPLv3+)](#open-source-license-agplv3)
- [Commercial License](#commercial-license)
- [Dependency Licenses](#dependency-licenses)
- [TensorRT Special Case](#tensorrt-special-case)
- [HuggingFace Models](#huggingface-models)
- [Choosing a License](#choosing-a-license)
- [FAQ](#faq)

## Overview

PITA is dual-licensed to support both open source and commercial use cases:

| License Type | Best For | Source Code Disclosure | Cost |
|--------------|----------|------------------------|------|
| **AGPLv3+** | Open source projects, research, evaluation | Required (including for network services) | Free |
| **Commercial** | Proprietary software, SaaS without source disclosure | Not required | Paid |

## Dual Licensing Model

### What is Dual Licensing?

Dual licensing means the same software is available under two different license options. You choose which license to use based on your needs.

### How It Works

1. **COBI, Inc. owns the copyright** to PITA
2. **All dependencies use permissive licenses** (MIT, BSD, Apache 2.0) that allow dual licensing
3. **You choose** between AGPLv3+ (free, open source) or Commercial (paid, proprietary)
4. **Your choice applies** to your use of PITA - it doesn't affect others

### Legal Basis

This dual licensing model is legally sound because:
- All third-party dependencies use permissive licenses compatible with both AGPLv3 and proprietary use
- No copyleft dependencies (GPL/LGPL) restrict our licensing options
- COBI, Inc. owns the copyright and can license under multiple terms

## Open Source License (AGPLv3+)

### What is AGPLv3?

The **GNU Affero General Public License v3.0** is a strong copyleft license designed for network services. It's based on GPLv3 with an additional requirement for network use.

### Key Requirements

If you use PITA under AGPLv3+, you must:

1. **Provide source code** to users, including:
   - The PITA library code
   - Any modifications you make
   - Your application code that uses PITA

2. **Network use clause** (the "Affero" part):
   - If users interact with your software over a network (e.g., web API, SaaS)
   - You must offer them the source code
   - Even if you don't distribute the software

3. **License derivatives under AGPLv3+**:
   - Any software you create using PITA must also be AGPLv3+
   - You cannot create proprietary derivatives

4. **Include license notices**:
   - Keep copyright notices intact
   - Include the full AGPLv3 license text

### When to Use AGPLv3+

AGPLv3+ is perfect for:
- ✅ Open source projects
- ✅ Academic research
- ✅ Internal tools (no external users)
- ✅ Evaluation and testing
- ✅ Contributing back to the community

AGPLv3+ is NOT suitable for:
- ❌ Proprietary SaaS products
- ❌ Closed-source software
- ❌ Products where you can't disclose source code
- ❌ Integration into proprietary systems

### Compliance Checklist

To comply with AGPLv3+ when using PITA:

- [ ] Keep all copyright and license notices
- [ ] Include LICENSE file in distributions
- [ ] Make source code available to users
- [ ] For network services: provide download link for complete source code
- [ ] License your application under AGPLv3+ or compatible license
- [ ] Document any modifications you make

### Full License Text

The complete AGPLv3 license is available at:
- In this repository: [../LICENSE](../LICENSE)
- Official source: https://www.gnu.org/licenses/agpl-3.0.txt

## Commercial License

### What is the Commercial License?

A proprietary license that allows you to use PITA in closed-source software without the source code disclosure requirements of AGPLv3.

### What You Get

With a commercial license:

1. **No source code disclosure required**
   - Keep your application code proprietary
   - No AGPLv3 network use obligations

2. **Flexible terms**
   - Negotiated based on your use case
   - Can include support and maintenance

3. **Legal certainty**
   - Clear proprietary use rights
   - No copyleft concerns

### When You Need Commercial License

You need a commercial license if:
- ✅ Building proprietary SaaS products
- ✅ Embedding in closed-source software
- ✅ Cannot disclose source code (IP protection, customer requirements, etc.)
- ✅ Want to avoid AGPLv3 compliance complexity

### How to Obtain

1. **Contact us**: sales@cobi-inc.com
2. **Discuss your use case**: We'll determine appropriate terms
3. **Review agreement**: See [../LICENSE-COMMERCIAL](../LICENSE-COMMERCIAL) for template
4. **Execute license**: Sign agreement and pay licensing fee
5. **Receive license**: Written authorization to use under commercial terms

### Pricing

Pricing is customized based on:
- Number of users/deployments
- Commercial vs. internal use
- Support and maintenance requirements
- Organization size

Contact sales@cobi-inc.com for a quote.

## Dependency Licenses

PITA depends on various third-party libraries. All use **permissive licenses** that are compatible with both open source and commercial use.

### License Summary

| License | Count | Compatibility |
|---------|-------|---------------|
| **MIT** | 5 packages | ✅ Fully compatible with dual licensing |
| **BSD 3-Clause** | 7 packages | ✅ Fully compatible with dual licensing |
| **Apache 2.0** | 7 packages | ✅ Fully compatible with dual licensing |

### Complete List

See [../NOTICE](../NOTICE) for complete attribution information including:
- All dependency names and versions
- Copyright notices
- License texts

### What This Means for You

**Good news**: All dependencies are permissively licensed, so:
- ✅ No additional copyleft obligations
- ✅ Can use in proprietary software (with commercial PITA license)
- ✅ Simple compliance - just maintain attribution notices
- ✅ No conflicts with dual licensing model

**Your obligations**:
- Include the NOTICE file in distributions
- Maintain copyright notices
- For Apache 2.0 components: document any modifications

## TensorRT Special Case

### The Situation

The optional TensorRT backend uses NVIDIA's proprietary TensorRT library, which has **separate licensing requirements**.

### NVIDIA TensorRT License

TensorRT is proprietary software with restrictions:
- Cannot redistribute TensorRT itself
- Cannot create derivative works of TensorRT
- License states it cannot be "subjected to open source licenses"
- Users must obtain TensorRT directly from NVIDIA

### Potential AGPLv3 Interaction

There's a theoretical question: Does using AGPLv3-licensed PITA with proprietary TensorRT create a conflict?

**Arguments it's fine** (stronger position):
1. **Separate components**: TensorRT isn't part of PITA - users install it separately
2. **Mere aggregation**: AGPLv3 allows separate programs on same system
3. **Optional dependency**: TensorRT isn't required for PITA to function
4. **Dynamic linking**: No static linking or code integration
5. **Industry precedent**: Many AGPLv3 projects use proprietary databases, drivers, etc.

**Why we document it**:
- NVIDIA's license has unusually explicit restrictions
- AGPLv3's network clause is stricter than standard GPL
- Better to inform users than ignore the question

### Our Recommendation

**For most users**: This is likely fine under "mere aggregation" doctrine, similar to:
- AGPLv3 apps using proprietary databases (Oracle, SQL Server)
- AGPLv3 code using proprietary GPU drivers
- AGPLv3 software on Windows

**If you're concerned**:
1. Use vLLM or llama.cpp backends instead (both open source)
2. Consult with your legal counsel
3. Obtain PITA commercial license (removes AGPLv3 question entirely)

**If using TensorRT**:
- Install TensorRT separately from NVIDIA
- Accept NVIDIA's license terms
- See [TENSORRT-LICENSE-NOTICE.md](TENSORRT-LICENSE-NOTICE.md) for details

### For Commercial License Users

If you have a PITA commercial license:
- Still need to obtain TensorRT from NVIDIA separately
- Same NVIDIA license requirements apply
- But no AGPLv3 interaction question

## HuggingFace Models

### The Issue

While the HuggingFace libraries (transformers, datasets, huggingface-hub) are Apache 2.0 licensed, **individual models and datasets have their own licenses**.

### Model License Variety

Models on HuggingFace Hub may be:
- ✅ Permissively licensed (MIT, Apache 2.0, etc.)
- ⚠️ Copyleft licensed (GPL, AGPLv3, etc.)
- ⚠️ Non-commercial only
- ⚠️ Proprietary/custom licenses
- ⚠️ No clear license

### Your Responsibility

**Users must verify model licenses before use.**

PITA's license doesn't grant rights to models - those come from model authors.

### How to Check Model License

1. Visit model page on HuggingFace Hub
2. Look for "License" field in model card
3. Read any license files in the model repository
4. Check for usage restrictions

### Recommendations

**For open source projects**:
- Verify model license is AGPLv3-compatible
- Prefer permissively licensed models

**For commercial use**:
- Ensure model allows commercial use
- Verify no non-commercial restrictions
- Consider contacting model author for licensing

**Keep records**:
- Document which models you use
- Save copies of license information
- Update if you change models

## Choosing a License

### Decision Tree

```
Are you building proprietary/closed-source software?
├─ YES → Commercial License
└─ NO → Continue...

Will users access your software over a network (SaaS, web API)?
├─ YES → Can you provide source code to users?
│   ├─ YES → AGPLv3+ is fine
│   └─ NO → Commercial License
└─ NO → AGPLv3+ is fine

Do you plan to keep modifications proprietary?
├─ YES → Commercial License
└─ NO → AGPLv3+ is fine
```

### Comparison Table

| Feature | AGPLv3+ | Commercial |
|---------|---------|------------|
| **Cost** | Free | Paid |
| **Source disclosure** | Required (including network use) | Optional |
| **Derivative licensing** | Must be AGPLv3+ | Your choice |
| **Commercial use** | Allowed (with source disclosure) | Allowed |
| **Support** | Community | Available (negotiable) |
| **Best for** | Open source, research | Proprietary products |

### Still Unsure?

Contact us: sales@cobi-inc.com

We can help determine the best licensing option for your use case.

## FAQ

### General Questions

**Q: Can I use PITA for free?**
A: Yes, under the AGPLv3+ license. You must comply with AGPLv3 requirements (mainly source code disclosure).

**Q: Can I use PITA in commercial products?**
A: Yes, either under AGPLv3+ (with source disclosure) or with a commercial license (without source disclosure).

**Q: Can I evaluate PITA before deciding on a license?**
A: Yes! Use it under AGPLv3+ for evaluation. Contact us about commercial licensing when you're ready to deploy.

### AGPLv3+ Questions

**Q: What exactly do I need to disclose under AGPLv3?**
A: Your complete application source code, including modifications to PITA and any code that uses PITA.

**Q: Does AGPLv3 apply if I only use PITA internally?**
A: If there are no external users (just employees), source disclosure isn't required. But the code is still AGPLv3-licensed.

**Q: Can I use AGPLv3 PITA in a web API?**
A: Yes, but you must offer source code to API users. This is the key AGPLv3 "network use" requirement.

**Q: What if I just use PITA as-is without modifications?**
A: You still need to provide your application source code to users under AGPLv3.

### Commercial License Questions

**Q: How much does a commercial license cost?**
A: Pricing is customized. Contact sales@cobi-inc.com for a quote.

**Q: Can I get support with a commercial license?**
A: Support and maintenance can be included in commercial license terms.

**Q: Does commercial license cover future versions?**
A: Terms specify which versions are covered. Typically negotiated as part of agreement.

**Q: Can I switch from AGPLv3 to commercial later?**
A: Yes. Obtain a commercial license and your future use will be under commercial terms.

### Dependency Questions

**Q: Do I need to worry about dependency licenses?**
A: All dependencies are permissively licensed. Just include the NOTICE file and maintain attributions.

**Q: Can I use PITA's permissive dependencies in my proprietary code?**
A: Yes (with commercial PITA license). The dependencies allow this. But PITA itself still requires appropriate licensing.

### TensorRT Questions

**Q: Can I use TensorRT with AGPLv3 PITA?**
A: Likely yes, under "mere aggregation" - similar to using proprietary databases. See [TENSORRT-LICENSE-NOTICE.md](TENSORRT-LICENSE-NOTICE.md) for details. If concerned, consult legal counsel or get commercial license.

**Q: Do I need to pay NVIDIA for TensorRT?**
A: TensorRT has its own licensing from NVIDIA. Check NVIDIA's terms for your use case.

**Q: Does PITA commercial license include TensorRT?**
A: No. TensorRT must be licensed separately from NVIDIA regardless of your PITA license.

### Model Questions

**Q: Are HuggingFace models included in PITA's license?**
A: No. Models have separate licenses from their authors. You must verify model licenses independently.

**Q: Can I use any model with commercial PITA license?**
A: Only if the model's license allows your use case. Check each model's license.

**Q: Where do I find model license information?**
A: On the model's HuggingFace Hub page, in the "License" field and any license files in the repository.

## Contact

For licensing questions, commercial licensing inquiries, or legal clarifications:

**Email**: sales@cobi-inc.com
**Repository**: https://github.com/cobi-inc-MC/pita
**Issues**: https://github.com/cobi-inc-MC/pita/issues

---

**Disclaimer**: This guide provides general information about PITA's licensing. It is not legal advice. For specific legal questions about your use case, consult with qualified legal counsel.

**Last Updated**: January 2026
**PITA Version**: 0.0.1
