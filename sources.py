FRED = "https://fred.stlouisfed.org"


def _fred(sid, title, publisher, **kw):
    cite = f"{publisher}, {title} [{sid}], retrieved from FRED, Federal Reserve Bank of St. Louis; {FRED}/series/{sid}."
    return dict(series=sid, title=title, publisher=publisher + " (via FRED, Federal Reserve Bank of St. Louis)", page=f"{FRED}/series/{sid}",
                csv=f"{FRED}/graph/fredgraph.csv?id={sid}", citation=cite, fmt="CSV, two columns: date and value (FRED download)", **kw)


BLS, EIA, IMF = "U.S. Bureau of Labor Statistics", "U.S. Energy Information Administration", "International Monetary Fund"

MARKETS = {
    "Tomatoes": _fred("APU0000712311", "Average Price: Tomatoes, Field Grown (Cost per Pound/453.6 Grams) in U.S. City Average", BLS,
                      kind="Fresh produce", frequency="Monthly", unit="lb", per="per lb", currency="$", noun="lb", qty=1000, hold=0.06,
                      hold_why="spoilage and storage", now="Sell now", ing="selling",
                      why="Retail price collected by BLS in 75 urban areas. The standard monthly record for US tomato prices, back to 1980."),
    "Eggs": _fred("APU0000708111", "Average Price: Eggs, Grade A, Large (Cost per Dozen) in U.S. City Average", BLS,
                  kind="Perishable food", frequency="Monthly", unit="dozen", per="per dozen", currency="$", noun="dozen", qty=500, hold=0.03,
                  hold_why="spoilage and storage", now="Sell now", ing="selling",
                  why="Retail price collected by BLS. Shows sharp supply shocks, such as bird flu outbreaks."),
    "Ground beef": _fred("APU0000703112", "Average Price: Ground Beef, 100% Beef (Cost per Pound/453.6 Grams) in U.S. City Average", BLS,
                         kind="Meat", frequency="Monthly", unit="lb", per="per lb", currency="$", noun="lb", qty=500, hold=0.025,
                         hold_why="cold storage and shrinkage", now="Sell now", ing="selling",
                         why="Retail price collected by BLS. Follows the multi-year cattle cycle."),
    "Gasoline": _fred("GASREGW", "Weekly U.S. Regular All Formulations Retail Gasoline Prices (Dollars per Gallon)", EIA,
                      kind="Energy", frequency="Weekly (converted to monthly)", unit="gallon", per="per gallon", currency="$", noun="gallon", qty=10000, hold=0.006,
                      hold_why="storage and tied-up cash", now="Sell now", ing="selling",
                      why="Weekly pump price survey by the US EIA, back to 1990. The app averages it to months."),
    "Wheat": _fred("PWHEAMTUSDM", "Global price of Wheat (US No.1 Hard Red Winter, FOB Gulf)", IMF,
                   kind="Grain (world market)", frequency="Monthly", unit="metric ton", per="per metric ton", currency="$", noun="ton", qty=100, hold=0.012,
                   hold_why="storage, shrink and interest", now="Sell now", ing="selling",
                   why="IMF benchmark export price for wheat, the world reference for grain traders."),
    "Copper": _fred("PCOPPUSDM", "Global price of Copper (LME grade A cathode)", IMF,
                    kind="Industrial metal (world market)", frequency="Monthly", unit="metric ton", per="per metric ton", currency="$", noun="ton", qty=25, hold=0.006,
                    hold_why="interest and storage", now="Sell now", ing="selling",
                    why="IMF monthly average of LME copper prices, the world reference for metal traders."),
}
CPI = _fred("CPIAUCSL", "Consumer Price Index for All Urban Consumers: All Items in U.S. City Average", BLS, kind="Inflation", frequency="Monthly")
CUSTOM = dict(series="custom", title="Your own price file", publisher="You", kind="Custom market", frequency="Monthly or weekly", unit="unit", per="per unit", currency="$",
              noun="unit", qty=100, hold=0.01, hold_why="storage and tied-up cash", now="Sell now", ing="selling", page="", csv="", citation="",
              fmt="CSV or Excel with a date column and a price column", why="Any price history with at least 5 years of monthly (or weekly) data.")
