![Python](https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/status-active-brightgreen)

## CreditRisk: Non-Linear Loan Profitability & Portfolio Modeling

Live Link: https://creditportfolio-profit.streamlit.app/

An interactive deployment of a non-linear credit scoring model better modeling risk than traditional Logistic Regression.
> User friendly

### Overview:

Information Avaliable:
- Maximum portfolio profit
- Profit maximising probability of default cutoff
- Client's probability of default
- Suggested interest rate for the client's risk profile
- The breakeven interest rate to charge them
- Expected annual profit from the client
- What features lead to the client's probability of default

### Tech Stack:

Language: Python 3.13

Frontend: Streamlit

Data Analysis: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Host: Streamlit Community Cloud

### Financial Maths:

Expected profit: $$E[P] = ((1 - PD) \cdot r \cdot EAD) - (PD \cdot LGD \cdot EAD)$$

Breakeven interest rate: $$r_b = \frac{LGD \cdot PD}{1 - PD}$$

Interest rate (r): $$r = \text{Prime Rate} + \text{Risk Premium}$$

### Installation

1. Clone the repository:
```bash
git clone https://github.com/james-sheridan-code/credit_portfolio.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run streamlit_app.py
```

