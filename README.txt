# PriceWise ⚖️
**A Pricing & Timing Tool for Small Sellers**

### What is this?
Pricing is a headache. Whether you're selling wheat, handmade tables, or renting out an apartment, you usually have two questions:
1. "Am I charging a fair price?" (Or am I ripping myself off?)
2. "Should I sell now, or wait for the price to go up?"

PriceWise answers those questions. Instead of showing you a bunch of confusing charts and making you guess, it uses two small neural networks to crunch the numbers and give you a straight answer: "Underpriced," "Sell Now," or "Wait."

---

### How to Run It

1. **Get the libraries**
   Open your terminal and install the basics:
   pip install -r requirements.txt

2. **Start the app**
   Run the main file:
   streamlit run app.py

   (This should pop open a browser window automatically.)

3. **Check the math (Optional)**
   If you want to make sure the AI isn't hallucinating, run the tests:
   python tests/test_core.py

---

### What's Inside?

Here’s a quick breakdown of the files so you know where to look:

*   **app.py**: The main interface. This handles the buttons, inputs, and draws the charts.
*   **simulator.py**: Since we don't have a live feed of USDA crop prices or Zillow rent data yet, this script generates "fake" history that behaves like the real world (seasons, inflation, random shocks).
*   **fair_price_net.py**: The first brain. It looks at costs + supply + demand to figure out what the "Fair" price actually is.
*   **trend_net.py**: The second brain. It looks at the last few months of history to guess if prices are going up or down next month.
*   **advisor.py**: The translator. It takes the complicated math from the AI and turns it into plain English advice like "Risk: High probability of being undercut."
*   **theme.py**: Just the styling settings to make it look clean and professional (no default Streamlit visuals).

---

### Real Talk (Limitations)

Right now, PriceWise runs on **simulated data**.

I built `simulator.py` to be realistic—prices go up when demand is high, they crash when there's an oversupply, and they follow seasonal trends—but it is **not** pulling live market data from the internet.

To make this a "real" product, you'd just need to swap out `simulator.py` for a script that scrapes real pricing data (like from Etsy, Zillow, or a commodities API). The AI models wouldn't even need to change; you'd just retrain them on the real CSV files.
