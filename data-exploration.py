import tensorflow as tf
import random
import pandas as pd
import numpy as np

mushroom_data = pd.read_csv('/content/mushrooms.csv')

mushroom_data[:5]

titles = ['Mushroom', 'capShape', 'capSurface', 'capColor', 'bruises', 'odor', 'gillAttachment', 'gillSpacing', 'gillSize', 'gillColor', 'stalkShape', 'stalkRoot', 'stalkSurfaceAboveRing', 'stalkSurfaceBelowRing', 'stalkColorAboveRing', 'stalkColorBelowRing', 'veilType', 'veilColor', 'ringNumber', 'ringType', 'sporePrintColor', 'population', 'habitat']

def numerize_data():
  mushroom_data['Mushroom'] = pd.Categorical(mushroom_data['Mushroom'])
  mushroom_data['Mushroom'] = mushroom_data.Mushroom.cat.codes

  mushroom_data['capShape'] = pd.Categorical(mushroom_data['capShape'])
  mushroom_data['capShape'] = mushroom_data.capShape.cat.codes

  mushroom_data['capSurface'] = pd.Categorical(mushroom_data['capSurface'])
  mushroom_data['capSurface'] = mushroom_data.capSurface.cat.codes

  mushroom_data['capColor'] = pd.Categorical(mushroom_data['capColor'])
  mushroom_data['capColor'] = mushroom_data.capColor.cat.codes

  mushroom_data['bruises'] = pd.Categorical(mushroom_data['bruises'])
  mushroom_data['bruises'] = mushroom_data.bruises.cat.codes

  mushroom_data['odor'] = pd.Categorical(mushroom_data['odor'])
  mushroom_data['odor'] = mushroom_data.odor.cat.codes

  mushroom_data['gillAttachment'] = pd.Categorical(mushroom_data['gillAttachment'])
  mushroom_data['gillAttachment'] = mushroom_data.gillAttachment.cat.codes

  mushroom_data['gillSpacing'] = pd.Categorical(mushroom_data['gillSpacing'])
  mushroom_data['gillSpacing'] = mushroom_data.gillSpacing.cat.codes

  mushroom_data['gillSize'] = pd.Categorical(mushroom_data['gillSize'])
  mushroom_data['gillSize'] = mushroom_data.gillSize.cat.codes

  mushroom_data['gillColor'] = pd.Categorical(mushroom_data['gillColor'])
  mushroom_data['gillColor'] = mushroom_data.gillColor.cat.codes

  mushroom_data['stalkShape'] = pd.Categorical(mushroom_data['stalkShape'])
  mushroom_data['stalkShape'] = mushroom_data.stalkShape.cat.codes

  mushroom_data['stalkRoot'] = pd.Categorical(mushroom_data['stalkRoot'])
  mushroom_data['stalkRoot'] = mushroom_data.stalkRoot.cat.codes

  mushroom_data['stalkSurfaceAboveRing'] = pd.Categorical(mushroom_data['stalkSurfaceAboveRing'])
  mushroom_data['stalkSurfaceAboveRing'] = mushroom_data.stalkSurfaceAboveRing.cat.codes

  mushroom_data['stalkSurfaceBelowRing'] = pd.Categorical(mushroom_data['stalkSurfaceBelowRing'])
  mushroom_data['stalkSurfaceBelowRing'] = mushroom_data.stalkSurfaceBelowRing.cat.codes

  mushroom_data['stalkColorAboveRing'] = pd.Categorical(mushroom_data['stalkColorAboveRing'])
  mushroom_data['stalkColorAboveRing'] = mushroom_data.stalkColorAboveRing.cat.codes

  mushroom_data['stalkColorBelowRing'] = pd.Categorical(mushroom_data['stalkColorBelowRing'])
  mushroom_data['stalkColorBelowRing'] = mushroom_data.stalkColorBelowRing.cat.codes

  mushroom_data['veilType'] = pd.Categorical(mushroom_data['veilType'])
  mushroom_data['veilType'] = mushroom_data.veilType.cat.codes

  mushroom_data['veilColor'] = pd.Categorical(mushroom_data['veilColor'])
  mushroom_data['veilColor'] = mushroom_data.veilColor.cat.codes

  mushroom_data['ringNumber'] = pd.Categorical(mushroom_data['ringNumber'])
  mushroom_data['ringNumber'] = mushroom_data.ringNumber.cat.codes

  mushroom_data['ringType'] = pd.Categorical(mushroom_data['ringType'])
  mushroom_data['ringType'] = mushroom_data.ringType.cat.codes

  mushroom_data['sporePrintColor'] = pd.Categorical(mushroom_data['sporePrintColor'])
  mushroom_data['sporePrintColor'] = mushroom_data.sporePrintColor.cat.codes

  mushroom_data['population'] = pd.Categorical(mushroom_data['population'])
  mushroom_data['population'] = mushroom_data.population.cat.codes

  mushroom_data['habitat'] = pd.Categorical(mushroom_data['habitat'])
  mushroom_data['habitat'] = mushroom_data.habitat.cat.codes
 
numerize_data()

mushroom_data[:5]

mushroom_data = tf.one_hot(mushroom_data, depth=len(titles))

mushroom_data[:5]

