# SamProject

##Install

```
virtualenv venv
source venv/bin/activate
pip install -r requirement.txt --allow-all-external
```

##Run Visualisation App

```
sudo python app.py
0.0.0.0/8000
```

##Run Neural Network
Run in the main project directory
```
python neural_net.py
```
The neural network can be changed, located in neuralNets directory

##Experiments

Located at: ./helpers/database/experiments

## Change the control and display:

- these are the parameters required for the visualisation:
DOMselector, b64_imageArray,
ControlX_coords, ControlY_coords,
ControlNumber_coords, Control_Colour,
dataLabels, tsneData,
sensible_number_of_points

Example:
```
function meta_vis() {

  var meta_visd = new metaVis("#meta_vis", mnist_images,
                  DATA.LAYER, DATA.EPOCH,
                  DATA.LAYER.length, DATA.LAYER, 
                  DATA.TSNE_LABELS, DATA.TSNE_DATA, 500);
  return meta_visd;
}
```
