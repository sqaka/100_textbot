## notice
almost all source code by Udemy: https://www.udemy.com/share/101WUgAkcTc1hWRH4=/


## readme
- please put base_text_data in app/utils/txt_data/

### using virtualenv
- `python3 -m venv .venv` in 100_textbot
- `source .venv/bin/activate`
- `pip install --upgrade pip` 
- `pip install -r requirements.txt`
- run `app/utils/prepare_text.py` and get .txt & .pickle files
- run `app/train_text.py` to make models

### using docker
- `docker-compose -f docker-compose.yaml up -d` in 100_textbot
- `docker-compose exec python3 bash`
- run `app/utils/prepare_text.py` and get .txt & .pickle files
- run `app/train_text.py` to make models

## directry
```
100_textbot
├── Dockerfile
|── docker-compose.yaml
|── requirements.txt
└── app
    |── train_txt.py
    |── test_models.py
    |── (encoder_model.h5)
    |── (decoder_model.h5)
    └── utils
        |── prepare_text.py
        |── make_files.py
        |── (prepare_text.txt)
        |── (prepare_text.pickle)
        └── txt_data
            |── (please put base_text_data in here)
            └── chars.txt
```