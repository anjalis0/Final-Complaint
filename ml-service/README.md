## custom-ml-model

## create python virtual env
```sh
python3 -m venv .venv
```

## activate the virtual env
```sh
.\.venv\Scripts\activate
```

## install required packages

```sh
pip install -r requirements.txt
```


### Start the server
Run the following command on from ther terminal, api will be running on port 8000
```sh
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
