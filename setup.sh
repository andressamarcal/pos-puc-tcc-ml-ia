mkdir -p ~/.streamlit/

pip3 install --upgrade pip
pip3 install -r requiriments.txt

echo "\
[general]\n\
email = \"andressagb@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml