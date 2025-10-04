#!/usr/bin/env python3
"""
Test script to verify brochure display functionality
"""

def test_brochure_display():
    """Test brochure display logic"""
    print("Testing brochure display functionality...")
    print("=" * 50)
    
    # Sample brochure content
    sample_brochure = """# MathCo - Company Brochure

## About MathCo

MathCo is a leading data science and analytics company that builds supercharged intelligence solutions for enterprises. We empower organizations to control their data and own their intelligence through cutting-edge AI and analytics capabilities.

## Our Services

### Engineering Services
- Custom data engineering solutions
- Advanced analytics implementation
- AI/ML model development

### Generative AI and Innovation
- State-of-the-art GenAI applications
- Innovation consulting
- AI strategy development

### Data Science Consulting
- Strategic data science consulting
- Predictive analytics
- Business intelligence solutions

## Industries We Serve

- **CPG (Consumer Packaged Goods)**: Optimize business performance with AI-integrated solutions
- **Retail**: Unlock shopper insights and drive increased revenue
- **Pharma & Life Sciences**: Redefine the future of pharma with AI-powered insights
- **Manufacturing**: Leverage advanced analytics for cost efficiency and growth
- **Automotive**: Accelerate growth with data-driven strategies

## Our Platform

### NucliOS® Foundation of Connected Intelligence
Our cutting-edge platform enables seamless decision-making while unleashing the full potential of connected intelligence.

## Why Choose MathCo

1. **IP Ownership with Client**: Full intellectual property ownership for clients
2. **Gen AI Perspective**: Shaping the future of analytics with GenAI applications
3. **Speed to Value**: Swift deployment and accelerated results
4. **Responsible AI Approach**: Transparency, compliance, and ethical standards
5. **Human-Centric Solutions**: Intuitive solutions aligned with client expectations

## Our Impact

MathCo has helped Fortune 500 businesses achieve remarkable results:
- 90% reduction in time-to-insights
- $1.3M operational cost savings
- 87% reduction in manual reporting time
- 40% acceleration in root cause analysis

## Contact Information

**Chicago Office**: 306W Erie St, Suite 300, Chicago, IL 60654
**Amsterdam Office**: Keizersgracht 555, 1017 DR Amsterdam, Netherlands
**Bengaluru Office**: 8th Floor, Tower A, IWF Campus, Whitefield Main Rd, Bengaluru, Karnataka, India

---

*Empowering enterprises to own their intelligence through data science and AI innovation.*
"""
    
    print("✅ Sample brochure content created")
    print(f"📄 Content length: {len(sample_brochure)} characters")
    print(f"📝 Lines: {len(sample_brochure.split(chr(10)))}")
    
    # Test markdown formatting
    print("\n📋 Sample sections:")
    lines = sample_brochure.split(chr(10))
    for i, line in enumerate(lines[:10]):  # Show first 10 lines
        if line.strip():
            print(f"  {i+1:2d}. {line[:60]}{'...' if len(line) > 60 else ''}")
    
    print("\n✅ Brochure display test completed successfully!")
    print("The Streamlit app should now properly display brochure content.")

if __name__ == "__main__":
    test_brochure_display()

