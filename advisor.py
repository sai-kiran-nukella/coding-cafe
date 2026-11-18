import math
import csv
import io

def calculate(quantity,quantity_unit,cost,price,price_unit,transport=0,other=0,fee=0,loss=0,days=0,storage=0):
    factors={'kg':('weight',1),'quintal':('weight',100),'tonne':('weight',1000),'litre':('volume',1),'dozen':('count',1)}
    if quantity_unit not in factors or price_unit not in factors or factors[quantity_unit][0]!=factors[price_unit][0]:raise ValueError('Quantity and price units must measure the same thing.')
    values=[quantity,cost,price,transport,other,fee,loss,days,storage]
    if any(not math.isfinite(v) or v<0 for v in values) or quantity<=0 or fee>=100 or loss>100:raise ValueError('Use positive quantity, non-negative amounts, fees below 100% and loss up to 100%.')
    base=quantity*factors[quantity_unit][1];sold=base*(1-loss/100);rate=price/factors[price_unit][1]
    revenue=sold*rate;purchase=base*cost;expenses=revenue*fee/100+transport+other+days*storage
    profit=revenue-purchase-expenses;be=(purchase+transport+other+days*storage)/(sold*(1-fee/100)) if sold else None
    if any(not math.isfinite(v) for v in [base,sold,revenue,purchase,expenses,profit]+([be] if be is not None else [])):raise ValueError('Amounts are too large.')
    return dict(quantity_base=base,saleable=sold,revenue=revenue,purchase=purchase,expenses=expenses,profit=profit,break_even=be,rate=rate)

def export_csv(rows):
    if not rows:return b''
    out=io.StringIO();w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader()
    for row in rows:
        w.writerow({k:("'"+v if isinstance(v,str) and bool(v) and v[:1] in '=+@-' else v) for k,v in row.items()})
    return out.getvalue().encode('utf-8-sig')
