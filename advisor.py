"""
Advisor Module
Acts as the 'Translator' between raw Neural Network outputs and human business strategy.
It converts probabilities and float values into actionable verdicts and risk assessments.
"""

def compare_to_fair(user_price, fair_price):
    """
    Analyzes the gap between User's Asking Price and the Model's Equilibrium Price.
    Returns a dictionary with status, plain-English reasoning, and specific tips.
    """
    # 1. Calculate the Gap
    if fair_price == 0: 
        fair_price = 0.01 # Prevent zero division
        
    diff = user_price - fair_price
    percent_diff = (diff / fair_price) * 100
    gap_abs = round(abs(percent_diff), 1)
    
    # 2. Determine Verdict based on sensitivity thresholds
    # We use tiers: Perfect (<2%), Fair (<6%), High/Low (6-15%), Extreme (>15%)
    
    if abs(percent_diff) <= 2.0:
        status = "Perfectly Aligned"
        reason = "Your price matches the theoretical market equilibrium exactly."
        tip = "Great pricing. Focus on marketing your quality to close the sale."
        
    elif abs(percent_diff) <= 6.0:
        status = "Fair"
        reason = f"You are within a normal trading range ({gap_abs}% variance)."
        tip = "This is a competitive price. You should see standard turnover."
        
    elif percent_diff > 6.0 and percent_diff <= 15.0:
        status = "Premium Pricing"
        reason = f"You are {gap_abs}% above the fair market value."
        tip = "Ensure your product has visible 'value-adds' (branding, service, quality) to justify this premium."
        
    elif percent_diff > 15.0:
        status = "Overpriced"
        reason = f"You are significantly ({gap_abs}%) above the market equilibrium."
        tip = "High risk of being undercut by competitors. Consider lowering price unless your product is unique."
        
    elif percent_diff < -6.0 and percent_diff >= -15.0:
        status = "Value Pricing"
        reason = f"You are {gap_abs}% below the fair market value."
        tip = "This will attract quick sales, but ensure you aren't eroding your profit margins unnecessarily."
        
    else: # percent_diff < -15.0
        status = "Underpriced"
        reason = f"You are {gap_abs}% below equilibrium. You may be leaving money on the table."
        tip = "Unless you need to liquidate cash immediately, consider raising your price."

    return {
        "status": status,
        "gap_percent": gap_abs,
        "reason": reason,
        "strategic_tip": tip,
        "fair_price": round(fair_price, 2)
    }

def sell_or_wait(trend_result):
    """
    Translates TrendNet technical signals into a binary business decision.
    Input: trend_result dict {direction, confidence, magnitude_percent}
    """
    direction = trend_result['direction']
    confidence = trend_result['confidence']
    magnitude = trend_result['magnitude_percent']
    
    # Defaults
    action = "Hold"
    reason = "Market signals are mixed."
    risk = "Standard market volatility applies."
    
    # 1. Logic for "Rise" (Bullish)
    if direction == "Rise":
        if confidence > 75:
            action = "Wait"
            reason = f"Strong upward momentum detected. Prices projected to gain ~{magnitude}% soon."
            risk = "Risk: Holding costs (storage/spoilage) must be lower than the expected ~{magnitude}% gain."
        elif confidence > 50:
            action = "Wait (Cautious)"
            reason = "Market is leaning upward, but signals are not definite."
            risk = "Risk: Trend is weak. Monitor closely; sell if momentum stalls."
        else:
            action = "List Normally"
            reason = "Slight upward bias, but insufficient confidence to delay sales."
            risk = "Risk: Noise in the market may mimic a trend."

    # 2. Logic for "Fall" (Bearish)
    elif direction == "Fall":
        if confidence > 75:
            action = "Sell Now"
            reason = f"Negative momentum detected. Prices likely to drop ~{magnitude}%."
            risk = "Risk: Selling now locks in value. Waiting risks further depreciation."
        elif confidence > 50:
            action = "Sell Soon"
            reason = "Market is softening. It is safer to secure liquidity now."
            risk = "Risk: Potential to miss a 'dead cat bounce' (temporary recovery), but downside risk is higher."
        else:
            action = "List Normally"
            reason = "Slight downward bias, but not enough to panic sell."
            risk = "Risk: Market is choppy."

    # 3. Logic for "Flat" (Stagnant)
    else:
        action = "List / Hold"
        reason = "Market is stable with low volatility."
        risk = "Risk: Capital tied up in inventory is not generating a return ('Opportunity Cost')."

    return {
        "action": action,
        "reason": reason,
        "risk_note": risk
    }
