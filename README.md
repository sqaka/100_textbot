## notice
almost all source code by Udemy: https://www.udemy.com/share/101WUgAkcTc1hWRH4=/


## readme
- please run docker in 100_textbot
- please put base text data in app/utils/txt_data/
- run app/utils/prepare_text.py and get .txt & .pickle files
- run app/train_text.py to make models

## directry
'''bash
100_textbot
├── Dockerfile
|── docker-compose.yaml
|── requirements.txt
└── app
    |── train_txt.py
    |── (some models output in here)
    └── utils
        |── prepare_text.py
        |── (prepare_text.txt)
        |── (prepare_text.pickle)
        └── txt_data
            |── (please put base_text_data in here)
            └── chars.txt
'''