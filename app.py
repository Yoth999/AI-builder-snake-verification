from fastai.vision.all import (
    load_learner,
    PILImage,
)

import streamlit as st
from PIL import Image

# set images
image1 = Image.open('logo1.jpg')


import gdown

file_id = '1jlQBNMnLXHlhLDq6eNdvTfrf6bvlaAmN'
output = 'model.pkl'
gdown.download(f'https://drive.google.com/uc?id={file_id}', output, quiet=False)
learn_inf = load_learner('model.pkl', cpu=True)

thaisnake=[
'gpv_venomous', 'mpv_venomous', 'pv_venomous', 'tv_venomous', 'krait_venomous', 'cobra_venomous', 'lgcoral_venomous', 'acoral_venomous'
    'achor_non-venomous', 'bronz_lnon-venomous', 'brown_lnon-venomous', 'collar_lnon-venomous', 'cat_lnon-venomous',
    'kukri_lnon-venomous', 'mocv_lnon-venomous', 'mud_lnon-venomous', 'sand_lnon-venomous', 'whip_lnon-venomous',
    'racer_lnon-venomous', 'tree_lnon-venomous', 'WaBs_lnon-venomous', 'cylin_non-venomous',
    'parea_non-venomous', 'pytho_non-venomous', 'typl_non-venomous', 'xeno_non-venomous'
]

engsnake=[
'gpv_venomous', 'mpv_venomous', 'pv_venomous', 'tv_venomous', 'krait_venomous', 'cobra_venomous', 'lgcoral_venomous', 'acoral_venomous'
    'achor_non-venomous', 'bronz_lnon-venomous', 'brown_lnon-venomous', 'collar_lnon-venomous', 'cat_lnon-venomous',
    'kukri_lnon-venomous', 'mocv_lnon-venomous', 'mud_lnon-venomous', 'sand_lnon-venomous', 'whip_lnon-venomous',
    'racer_lnon-venomous', 'tree_lnon-venomous', 'WaBs_lnon-venomous', 'cylin_non-venomous',
    'parea_non-venomous', 'pytho_non-venomous', 'typl_non-venomous', 'xeno_non-venomous'
]

def predict(img, learn):
    # make prediction
    pred, pred_idx, pred_prob = learn.predict(img)
    # Display the prediction
    thaisnakename = thaisnake[int(pred_idx)]
    engsnakename = engsnake[int(pred_idx)]
    st.success(f"This is {engsnakename} ({thaisnakename}) with the probability of {pred_prob[pred_idx]*100:.02f}%")
    # Display the test image
    # st.image(img, use_column_width=True)
    col1.image(img, use_column_width=True)
    # col2.image(''+image3)
    col2.image(Image.open('./thaimenu/'+str(int(pred_idx))+'.png'))

##################################
# Top Main
##################################
col1, col2 = st.beta_columns(2)

##################################
# Col1
##################################
col1.header("Your Food Image")

st.sidebar.image(image1)

fname = st.sidebar.file_uploader('Enter snake image to classify',type=['png', 'jpg', 'jpeg'],accept_multiple_files=False)
if fname is None:
    # fname = valid_images[0]
    col1.image(image2)
    col2.image(image3)
else :
    img = PILImage.create(fname)
    predict(img, learn_inf)

##################################
# sidebar
##################################
st.sidebar.markdown('This app, snakeidenAI, was developed by Yoth Srimanchanda and Pannawat Kerdpin as our project of AIbuilder x TSS class (ET32201).')
st.sidebar.write("snakeidenAI at Github [link](https://github.com/yoth99/AI-builder-snake-verification/)")

st.text(' ')
st.text(' ')

my_expander = st.beta_expander(label='End Credits Scene')
with my_expander:
    '[yoth](https://github.com/yoth999/) : Thank You so much for your help, tips, and advice because if it weren’t for you, my model wouldn’t be the way it is today.'
    '[pannawat](https://github.com/PK2301/),'
    '[AI Builders](https://www.youtube.com/@AIBuilders) : Thank You, AI Builders, for teaching me how to make this whole program; comparing my understanding of AI before I began the course compared to after the course is immeasurable.'
    'Google Colab : Thank you, Google team, for making Colab; because of Colab, I can train AI more efficiently.'
    'Family : I also would like to thankyou my family for giving me support during this time.'
    'This project was done without the use of Stack Overflow somehow...'