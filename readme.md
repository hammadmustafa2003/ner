# Named Entity Recognition

This project uses a model trained on tenderflow keras to identify multiply identities as geographical locations, organizations, time and few others too.

For front end it uses simple html javascript and an api call to get the results from the model.


## How to run the project:

- If you don't have ner-model.model folder, that run all the cells of src.ipynb
- It will create ner-model.model folder and file.var.
- Next run main.py file using FastAPI.
- Copy the url of your api to line 88 of project.html and append "?model_input=" to it. For example if your api has ip address of 127.0.0.1 and port number 8000. The line should look like this:		"constreq='http://127.0.0.1:8000/?model_input='+encodeURIComponent(textconatiner.value);"
- Next open project.html and give input to it.

## Dependencies:

- Numpy
- Pandas
- Tenserflow.keras
- FastAPI
