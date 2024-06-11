Follow these step for your own installation:
1. Download data for stable reproduce at [here](https://drive.google.com/drive/folders/1lIAFTlqL7TpaRqjiUGJZJgzsR7IqS28F?usp=drive_link) and paste as:
```
<location>
|---app.py
|---create_vector_db.py
|---...
|---data
    |---dbv3
    |---mdoel
    |---...
```
2. Setup environment:
```
conda create --name vcbchat python=3.8
conda activate vcbchat
pip install -r requirements.txt
```

3. Run the code as local:
- If you want to return the DB from the csv file:
```
python create_vector_db.py --embedding_path <your embedding model> --csv_path <path of original DB>  --vector_db_path <your DB output dir>
```
- If the installation of DB or you run the above code, run this for turn on the model and UI:
```
python app.py --model_path <your LLM model> --embedding_path <your embedding model> --vector_db_path <your DB> --csv_path <path of original DB> --debug <Turn on debug mode (1) or off (0)> --history <Turn on history feature (1) or off (0)>
```
- If the folder installation is complete exactly the same as the folder organization, run this as simple demo:
```
python app.py
```