## CreditRisk: Non-Linear Loan Profitability & Portfolio Modeling

An interactive deployment of a non-linear credit scoring model better modeling risk than traditional Logistic Regression.

### Overview:
This program allows for a simple and user friendly way to get information regarding individual and portfolio profits using a non-linear model.

Information Avaliable:
- Maximum portfolio profit
- Profit maximising probability of default cutoff
- Client's probability of default
- Suggested interest rate for the client's risk profile
- The breakeven interest rate to charge them
- Expected annual profit from the client
- What features lead to the client's probability of default

### Key Features:
- Simple dropdown options for client features
- Sliders for the Loss Given Default
- Interactive graphs for portfolio profit and client's features

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

