import os
import numpy as np
from PIL import Image
import skimage.filters as skf
import skimage.morphology as skm
import mahotas as mht

def procesar_imagen(img_path):
    """Este metodo procesa la imagen que se pasa como ruta.
       
       El procesamiento realiza el realzado arterial y posteriormente
       la binarizacion mediante el metodo de Otsu.
       
       Finalmente, se extrae el esqueleto arterial de la imagen
       previamente binarizada.

    Args:
        img_path (_type_): La ruta de la imagen a procesar.

    Returns:
       np.array, ...: Devuelve 5 matrices de tipo numpy.array,
                      las cuales representan:
                      1. La imagen original con valores enteros en el
                         rango [0, 255].
                      2. La imagen original con valores normalizados.
                      3. La imagen con la respuesta normalizada 
                         del metodo de Frangi usado para el
                         realzado arterial
                      4. La imagen binarizada de la respuesta normalizada
                         del metodo de Frangi. 
                         El umbral de binarizacion se calculo con el metodo de Otsu.
                      5. La imagen del esqueleto arterial, extraido de la imagen
                         binarizada.
    """
    
    # Se carga la imagen:
    img = Image.open(img_path)    
    # Se traslada a una matriz:
    img_arr_int = np.asarray(img.getdata(), dtype=np.uint8).reshape(img.size[0], img.size[1])
    # Se debe normalizar la imagen:
    img_arr_norm = img_arr_int / 255.0
    
    # Se realzan las arterias con el metodo de Frangi:
    sigmas = np.arange(1, 12, 0.5)
    beta_one = 0.5
    beta_two = 15
    img_vessel_raw = skf.frangi(img_arr_norm, sigmas=sigmas, alpha=beta_one, beta=beta_two, mode='wrap')
    
    # Se segmentan las arterias binarizando la imagen
    # con el método de Otsu:
    threshold = skf.threshold_otsu(img_vessel_raw)
    img_vessel_bin = img_vessel_raw > threshold
    
    # Se extrae el esqueleto arterial:
    img_vessel_skel = skm.skeletonize(img_vessel_bin)
    
    # Se devuelven las imagenes procesadas:
    return img_arr_int, img_arr_norm, img_vessel_raw, img_vessel_bin, img_vessel_skel

def extraer_intensidad(img_arr_norm):
    pix_min = np.min(img_arr_norm)
    pix_max = np.max(img_arr_norm)
    pix_median = np.median(img_arr_norm)
    pix_mean = np.mean(img_arr_norm)
    pix_variance = np.var(img_arr_norm)
    pix_stddev = np.std(img_arr_norm)
    
    intensity_feats=np.array([
                        pix_min.item(), pix_max.item(), pix_median.item(),
                        pix_mean.item(), pix_variance.item(), pix_stddev.item()
                    ])
    feats_name =    [
                        'Min', 'Max', 'Median', 
                        'Mean', 'Variance', 'StandardDeviation'
                    ]
    
    return intensity_feats, feats_name

def extraer_textura(img_arr_int):
    text_feats = mht.features.haralick(img_arr_int, return_mean=True)
    feats_name =[
                    'AngularSecondMoment',
                    'Contrast',
                    'Correlation',
                    'Variance',
                    'Homogeneity',
                    'SumAverage',
                    'SumVariance',
                    'SumEntropy',
                    'Entropy',
                    'DifferenceVariance',
                    'DifferenceEntropy',
                    'InformationMeasureOfCorrelation1',
                    'InformationMeasureOfCorrelation2'
                ]
    return text_feats, feats_name

def extraer_morfologia(img_arr_norm, img_vessel_bin, img_vessel_skel):
    number_of_vessel_pixels = np.sum(img_vessel_bin)
    vessel_density = np.sum(img_vessel_bin*1.0) / (img_vessel_bin.shape[0] * img_vessel_bin.shape[1])
    sum_of_vessels_length = np.sum(img_vessel_skel)
    
    vessel_mean_grays = np.mean(img_arr_norm[img_vessel_bin > 0])
    gray_level_coefficient_variation = np.std(img_arr_norm[img_vessel_bin > 0]) / vessel_mean_grays if vessel_mean_grays > 0 else 0
    
    morph_feats=np.array([
                    number_of_vessel_pixels.item(),
                    vessel_density.item(),
                    sum_of_vessels_length.item(),
                    gray_level_coefficient_variation.item()
                ])
    feats_name =[
                    'NumberOfVesselPixels',
                    'VesselDensity',
                    'SumOfVesselsLength',
                    'GrayLevelCoefficientVariation'
                ]
    
    return morph_feats, feats_name
                
def extraer_todo(ruta:str):
    headers = []
    features_dataset = None
    
    # Se enlistan los archivos dentro del directorio:
    files = os.listdir(ruta)
    
    # Se itera sobre cada elemento que apunta a un archivo:
    for i in range(0, len(files)):
        # Se establece la ruta completa del archivo:
        f = os.path.join(ruta, files[i])
        
        print("Procesando Archivo %d de %d: [%s]" % ((i+1), len(files), f))
        
        # Se procesa la imagen para hacer el realzado arterial, segmentar las arterias y extraer el esqueleto:
        img_arr_int, img_arr_norm, img_vessel_raw, img_vessel_bin, img_vessel_skel = procesar_imagen(f)
        
        # Se extraen las caracteristicas de intensidad, textura y morfologia: 
        intensity_feats, intens_feats_name = extraer_intensidad(img_arr_norm)
        text_feats, text_feats_name = extraer_textura(img_arr_int)
        morph_feats, morph_feats_name = extraer_morfologia(img_arr_norm, img_vessel_bin, img_vessel_skel)
        
        # Se genera un solo vector de caracteristicas con los vectores obtenidos previamente:
        feat_vector = np.hstack((intensity_feats, text_feats, morph_feats))
        
        # Se va generando el dataset. La primera vez, se genera el vector de encabezados.
        if i == 0:
            headers = intens_feats_name + text_feats_name + morph_feats_name
            features_dataset = feat_vector
        else :
            features_dataset = np.vstack((features_dataset, feat_vector))
    return features_dataset, headers
    
def __main__():
    # Se define el directorio que contiene las imagenes:    
    dir_working     = './DatasetECE2024/'
    #dir_working     = '/opt/lampp/htdocs/public/mgilr/ece2024/contents/WorkSpaceECE2024/DatasetECE2024/'
    dir_training    = dir_working + '01_Training/'
    dir_validation  = dir_working + '02_Validation/'
        
    # Se extraen las caracteristicas de las imagenes que se utilizaran
    # para el entrenamiento y se generan el dataset y los encabezados 
    # de las columnas:
    dataset_training, headers_training      = extraer_todo(dir_training)
    
    # Se extraen las caracteristicas de las imagenes que se utilizaran
    # para validacion y se generan el dataset y los encabezados 
    # de las columnas:
    dataset_validation, headers_validation  = extraer_todo(dir_validation)    
    
    # Se genera un encabezado para los archivos csv de entrenamiento y validacion:
    csv_header = ','.join(headers_training)
    
    # Se guardan los datasets como archivos de texto separados por coma:
    np.savetxt(dir_working + '01_Training.csv', dataset_training, delimiter=',', fmt='%f', header=csv_header, comments='')
    np.savetxt(dir_working + '02_Validation.csv', dataset_validation, delimiter=',', fmt='%f', header=csv_header, comments='')
    
    print('Data Files saved')

__main__()