"""
COMPARE EXISTING SUBMISSIONS
Compare Step 5 vs Step 6 vs Step 7 vs Step 8 predictions directly

This shows us IF the bug fix actually changed the predictions
"""

import pandas as pd
import numpy as np

print("="*80)
print("COMPARING SUBMISSION FILES")
print("="*80)

# Load all submissions
try:
    step5 = pd.read_csv('lightgbm_step5_no_calibration.csv')
    print("✅ Loaded Step 5 submission")
except:
    print("❌ Step 5 file not found")
    step5 = None

try:
    step6 = pd.read_csv('lightgbm_step6_minimal.csv')
    print("✅ Loaded Step 6 submission")
except:
    print("❌ Step 6 file not found")
    step6 = None

try:
    step7 = pd.read_csv('lightgbm_step7_pruned.csv')
    print("✅ Loaded Step 7 submission")
except:
    print("❌ Step 7 file not found")
    step7 = None

try:
    step8 = pd.read_csv('lightgbm_step8_bugfix.csv')
    print("✅ Loaded Step 8 submission")
except:
    print("❌ Step 8 file not found (run Step 8 first!)")
    step8 = None

print("\n" + "="*80)
print("PREDICTION STATISTICS")
print("="*80)

for name, df in [('Step 5', step5), ('Step 6', step6), ('Step 7', step7), ('Step 8', step8)]:
    if df is not None:
        print(f"\n{name}:")
        print(f"  Mean:        {df['predicted_weight'].mean():10,.0f} kg")
        print(f"  Median:      {df['predicted_weight'].median():10,.0f} kg")
        print(f"  Std:         {df['predicted_weight'].std():10,.0f} kg")
        print(f"  Max:         {df['predicted_weight'].max():10,.0f} kg")
        print(f"  Sum:         {df['predicted_weight'].sum():10,.0f} kg")
        print(f"  Pred > 0:    {(df['predicted_weight'] > 0).sum():10,} ({(df['predicted_weight'] > 0).sum() / len(df) * 100:5.1f}%)")
        print(f"  Pred == 0:   {(df['predicted_weight'] == 0).sum():10,} ({(df['predicted_weight'] == 0).sum() / len(df) * 100:5.1f}%)")

# Compare Step 5 vs Step 6
if step5 is not None and step6 is not None:
    print("\n" + "="*80)
    print("STEP 5 vs STEP 6 COMPARISON")
    print("="*80)
    
    diff = step6['predicted_weight'] - step5['predicted_weight']
    
    print(f"\nPrediction differences (Step 6 - Step 5):")
    print(f"  Mean diff:       {diff.mean():10,.0f} kg")
    print(f"  Max increase:    {diff.max():10,.0f} kg")
    print(f"  Max decrease:    {diff.min():10,.0f} kg")
    print(f"  Identical preds: {(diff == 0).sum():10,} ({(diff == 0).sum() / len(diff) * 100:5.1f}%)")
    print(f"  Changed preds:   {(diff != 0).sum():10,} ({(diff != 0).sum() / len(diff) * 100:5.1f}%)")
    
    # Check zeros
    step5_zeros = (step5['predicted_weight'] == 0).sum()
    step6_zeros = (step6['predicted_weight'] == 0).sum()
    
    print(f"\nZero predictions:")
    print(f"  Step 5: {step5_zeros:,} zeros")
    print(f"  Step 6: {step6_zeros:,} zeros")
    print(f"  Diff:   {step6_zeros - step5_zeros:+,} more zeros in Step 6")
    
    # Find which became zeros
    became_zero = (step5['predicted_weight'] > 0) & (step6['predicted_weight'] == 0)
    became_nonzero = (step5['predicted_weight'] == 0) & (step6['predicted_weight'] > 0)
    
    print(f"\nPredictions that changed:")
    print(f"  Became 0:     {became_zero.sum():,}")
    print(f"  Became non-0: {became_nonzero.sum():,}")

# Compare Step 6 vs Step 7
if step6 is not None and step7 is not None:
    print("\n" + "="*80)
    print("STEP 6 vs STEP 7 COMPARISON (removed features)")
    print("="*80)
    
    diff = step7['predicted_weight'] - step6['predicted_weight']
    
    print(f"\nPrediction differences (Step 7 - Step 6):")
    print(f"  Mean diff:       {diff.mean():10,.0f} kg")
    print(f"  Identical preds: {(diff == 0).sum():10,} ({(diff == 0).sum() / len(diff) * 100:5.1f}%)")
    print(f"  Changed preds:   {(diff != 0).sum():10,} ({(diff != 0).sum() / len(diff) * 100:5.1f}%)")
    
    if (diff == 0).sum() / len(diff) > 0.95:
        print("\n⚠️  WARNING: Step 6 and Step 7 are almost identical!")
        print("    This confirms BOTH have the same bug!")

# Compare Step 5 vs Step 8
if step5 is not None and step8 is not None:
    print("\n" + "="*80)
    print("STEP 5 vs STEP 8 COMPARISON (BUG FIX TEST)")
    print("="*80)
    
    diff = step8['predicted_weight'] - step5['predicted_weight']
    
    print(f"\nPrediction differences (Step 8 - Step 5):")
    print(f"  Mean diff:       {diff.mean():10,.0f} kg")
    print(f"  Max increase:    {diff.max():10,.0f} kg")
    print(f"  Max decrease:    {diff.min():10,.0f} kg")
    print(f"  Identical preds: {(diff == 0).sum():10,} ({(diff == 0).sum() / len(diff) * 100:5.1f}%)")
    print(f"  Changed preds:   {(diff != 0).sum():10,} ({(diff != 0).sum() / len(diff) * 100:5.1f}%)")
    
    # Check zeros
    step5_zeros = (step5['predicted_weight'] == 0).sum()
    step8_zeros = (step8['predicted_weight'] == 0).sum()
    
    print(f"\nZero predictions:")
    print(f"  Step 5: {step5_zeros:,} zeros")
    print(f"  Step 8: {step8_zeros:,} zeros")
    print(f"  Diff:   {step8_zeros - step5_zeros:+,}")
    
    # Find which became zeros
    became_zero = (step5['predicted_weight'] > 0) & (step8['predicted_weight'] == 0)
    became_nonzero = (step5['predicted_weight'] == 0) & (step8['predicted_weight'] > 0)
    
    print(f"\nPredictions that changed:")
    print(f"  Became 0:     {became_zero.sum():,}")
    print(f"  Became non-0: {became_nonzero.sum():,}")
    
    # Similarity score
    correlation = np.corrcoef(step5['predicted_weight'], step8['predicted_weight'])[0, 1]
    print(f"\nCorrelation between Step 5 and Step 8: {correlation:.6f}")
    
    if step5_zeros == step8_zeros:
        print("\n✅ GOOD: Step 5 and Step 8 have same number of zeros")
        print("   Bug fix may have worked!")
    else:
        print(f"\n⚠️  Step 8 has {abs(step8_zeros - step5_zeros)} different zeros than Step 5")

# Compare Step 7 vs Step 8
if step7 is not None and step8 is not None:
    print("\n" + "="*80)
    print("STEP 7 vs STEP 8 COMPARISON (only bug fix changed)")
    print("="*80)
    
    diff = step8['predicted_weight'] - step7['predicted_weight']
    
    print(f"\nPrediction differences (Step 8 - Step 7):")
    print(f"  Mean diff:       {diff.mean():10,.0f} kg")
    print(f"  Identical preds: {(diff == 0).sum():10,} ({(diff == 0).sum() / len(diff) * 100:5.1f}%)")
    print(f"  Changed preds:   {(diff != 0).sum():10,} ({(diff != 0).sum() / len(diff) * 100:5.1f}%)")
    
    # Check zeros
    step7_zeros = (step7['predicted_weight'] == 0).sum()
    step8_zeros = (step8['predicted_weight'] == 0).sum()
    
    print(f"\nZero predictions:")
    print(f"  Step 7: {step7_zeros:,} zeros")
    print(f"  Step 8: {step8_zeros:,} zeros")
    print(f"  Diff:   {step8_zeros - step7_zeros:+,}")
    
    if step7_zeros != step8_zeros:
        print("\n✅ GOOD: Step 8 has different number of zeros than Step 7")
        print("   Bug fix is having an effect!")
        
        became_nonzero = (step7['predicted_weight'] == 0) & (step8['predicted_weight'] > 0)
        print(f"\n   {became_nonzero.sum():,} predictions went from 0 to non-zero")
        print("   These are the RMs that were incorrectly zeroed by the bug!")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

print("\n💡 KEY INSIGHTS:")
print("   - If Step 6 and Step 7 are identical → same bug in both")
print("   - If Step 8 has fewer zeros than Step 6/7 → bug fix working!")
print("   - If Step 8 matches Step 5 closely → we're back on track!")

print("\n🎯 RECOMMENDATION:")
if step5 is not None and step8 is not None:
    step5_zeros = (step5['predicted_weight'] == 0).sum()
    step8_zeros = (step8['predicted_weight'] == 0).sum()
    
    if abs(step5_zeros - step8_zeros) < 100:
        print("   ✅ Step 8 looks very similar to Step 5")
        print("   ✅ SUBMIT Step 8 to Kaggle!")
        print("   ✅ Expected score: ~6,200-6,300 (close to Step 5's 6,236)")
    elif step8_zeros < step5_zeros:
        print("   ⚠️  Step 8 has fewer zeros than Step 5")
        print("   ⚠️  This is unexpected - may be over-predicting")
        print("   ⚠️  Review Step 8 logic before submitting")
    else:
        print("   ⚠️  Step 8 has more zeros than Step 5")
        print("   ⚠️  Bug may not be fully fixed")
        print("   ⚠️  Review the days_since_last logic")
else:
    print("   ⚠️  Need both Step 5 and Step 8 files to compare!")
    print("   ⚠️  Run Step 8 first, then re-run this script")

print("\n" + "="*80)