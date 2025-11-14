#!/bin/bash
echo "Layla trip:"
curl -s "http://127.0.0.1:8000/ask?q=When%20is%20Layla%20planning%20her%20trip%20to%20London%3F"
echo
echo "Vikram cars:"
curl -s "http://127.0.0.1:8000/ask?q=How%20many%20cars%20does%20Vikram%20Desai%20have%3F"
echo
echo "Amira restaurants:"
curl -s "http://127.0.0.1:8000/ask?q=What%20are%20Amira%27s%20favorite%20restaurants%3F"
