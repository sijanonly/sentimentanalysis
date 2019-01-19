# Django project with LSTM model for sentiment analysis

### For saving model after training

`torch.save({'model_state_dict': model.state_dict()}, FILE_PATH)`

### For loading model for inference
```
device = torch.device('cpu')
checkpoint = torch.load(FILE_PATH, map_location=device)
state_dict =checkpoint['model_state_dict']
model.load_state_dict(state_dict)
```

## Setup

* download model file : [file1](https://www.dropbox.com/s/c39m71h3ai0exrn/lstmmodelgpu2.tar?dl=1)

* download dictionary(word_to_int) mapper file : [file2](https://www.dropbox.com/s/az7ex7ezmhxp551/dict.pkl?dl=1)
* make sure these files are located at the level of `manage.py` file
* run `python manage.py runserver` and check `localhost:8000` 




