# gesture-recognition

This contains 3 main python files -
1) collectData.py which collects data from a video capture of size 320x240 i.e. 76800 neurons. Mixing it up with the number keys entered with the specific frame. This is then saved as 'training.npz' in the root folder.

2) training.py which trains the neural network currently consisting of 76800->64->32->4 layers. This then saves the xml file to the 'mlp_xml' folder. 

3) mainScript.py Which then finally takes frames and predicts the output.
