from worldbankapp import app
import os

# Use run for calling directly with python worldbank.py
# Comment out when utilizing gunicorn
# Also, switch Procfile to use gunicorn vs python command
port = int(os.environ.get('PORT', 33507))
app.run(host='0.0.0.0', port=port, debug=True)