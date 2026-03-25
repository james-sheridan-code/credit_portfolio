## Web-based platform for both portfolio wide analysis and individual loan expected outcomes

Overview:
This program allows for a simple and user friendly way to get information regarding individual and portfolio profits using a non-linear model that beats the standard logistic regression modelling of credit. 
You can see:
- Maximum portfolio profit
- Profit maximising probability of default cutoff
- Client's probability of default
- Suggested interest rate for the client's risk profile
- The breakeven interest rate to charge them
- Expected annual profit from the client
- What features lead to the client's probability of default

Key Features:
- Simple dropdown options for client features
- Sliders for the Loss Given Default
- Interactive graphs for portfolio profit and client's features

Tech Stack:
Language: Python 3.13
Frontend: Streamlit
Data Analysis: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Host: Streamlit Community Cloud

Financial Maths:
Expected profit = ((1 - PD) * r * EAD) - (PD * LDG * EAD)
Breakeven interest rate = (LGD * PD) / (1 - PD)
Interest rate (r) = prime_rate + risk_premium

Installation and Setup:
Clone the repo:
Bash
git clone https://github.com/yourusername/your-repo.git

Install dependencies:
Bash
pip install -r requirements.txt

Run the app:
Bash
streamlit run streamlit_app.py

