# instrument_cnn
####Parsear información####
Ejecutar py parser_data.py. 
INPUT: En carpeta DATA (piano y guitarra)
OUTPUT: Archivo json (data_piano.json)
####Entrenar y Testear a la red####
Ejecutar py RNN.py
INPUT: data_piano.json
OUTPUT: modelo-entrenado.h5 y gráfica con el accuracy y testing. 
####Requirements####
Tensorflow: 	pip install Tensorflow
Librosa: 	pip install Librosa
Scipy: 		pip install scipy
Numpy: 		pip install numpy
Matplotlib:	pip install matplotlib
Sklearn:	pip install sklearn