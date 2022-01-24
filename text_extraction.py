import streamlit as st
from PIL import Image
import requests
import pandas as pd
import os
import base64
import json
from joblib import Parallel, delayed
import datetime
import time
from stqdm import stqdm
from multiprocessing import Pool
import boto3
from io import StringIO
import logging
import collections


client = boto3.client('s3',
                      aws_access_key_id=st.secrets["access_key"],
                      aws_secret_access_key=st.secrets["secret_key"]
                      )

bucket = st.secrets["bucket"]
prefix = st.secrets["prefix"]


app_id = st.secrets["app_id"]
app_key = st.secrets["app_key"]


class TailLogHandler(logging.Handler):

    def __init__(self, log_queue):
        logging.Handler.__init__(self)
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.append(self.format(record))


class TailLogger(object):

    def __init__(self, maxlen):
        self._log_queue = collections.deque(maxlen=maxlen)
        self._log_handler = TailLogHandler(self._log_queue)

    def contents(self):
        return '\n'.join(self._log_queue)

    @property
    def log_handler(self):
        return self._log_handler

      
def mathpix_text_asciimath_textapi(url):
    tries = 3
    errored_files = []

    data = {
        "src": url,
        "formats": ["data", "text"],
        "data_options": {"include_asciimath": True},
        "include_smiles": True
    }
    
    for i in range(tries):
        try:
            r = requests.post("https://api.mathpix.com/v3/text",
                              json=data,
                              headers={
                                  "app_id": app_id,
                                  "app_key": app_key,
                                  "Content-type": "application/json"
                              })
            response = r.json()
        except Exception as e:
            if i < (tries - 1):
                continue
            else:
                errored_files.append(url)
                response = {}
                response["text"] = 'error'
                response["data"] = 'error'
                print(e)
                logger.error(f"Error Processing : {url} {e}")

    # return response['text'], response['data']
    return response

  
def replacement_function(text):
    text = text.replace('\\mathrm{~cm}', 'cm')
    text = text.replace('\\mathrm{~jam}', 'jam')
    text = text.replace('\\mathrm{~Rp.}', 'Rp.')
    text = text.replace('\\mathrm{~Hz}', 'Hz')
    text = text.replace('\\mathrm{~dm}', 'dm')
    text = text.replace('\\mathrm{~kg}', 'kg')
    text = text.replace('\\mathrm{~m}', 'm')
    text = text.replace('\\mathrm{cm}', 'cm')
    text = text.replace('\\mathrm{jam}', 'jam')
    text = text.replace('\\mathrm{Rp.}', 'Rp.')
    text = text.replace('\\mathrm{Hz}', 'Hz')
    text = text.replace('\\mathrm{dm}', 'dm')
    text = text.replace('\\mathrm{kg}', 'kg')
    text = text.replace('\\mathrm{P}', 'kg')
    text = text.replace('\\mathrm{m}', 'm')
    text = text.replace('\\mathrm{q}', 'q')
    text = text.replace('\\mathrm{r}', 'r')
    text = text.replace('<smiles>', '')
    text = text.replace('</smiles>', '')
    text = text.replace('\\Delta', 'segitiga')
    text = text.replace('\\triangle', 'segitiga')
    text = text.replace('\\angle', 'sudut')
    text = text.replace('\\pi', 'pi')
    text = text.replace('\\%', '%')
    text = text.replace('\\theta', 'theta')
    text = text.replace('\\gamma', 'gamma')
    text = text.replace('\\lambda', 'lambda')
    text = text.replace('\\beta', 'b')
    text = text.replace('\\alpha', 'a')
    text = text.replace('\\ell', 'l')
    text = text.replace('\\sin', 'sin')
    text = text.replace('\\cos', 'cos')
    text = text.replace('\\tan', 'tan')
    text = text.replace('\\sec', 'sec')
    text = text.replace('\\csc', 'csc')
    text = text.replace('\\cot', 'cot')
    text = text.replace('\\|dots', '....')
    text = text.replace('\\cdot', '.')
    text = text.replace('^{\circ}', '')
    text = text.replace('\\circ', 'o')
    text = text.replace('\\sqrt{x}', 'akar(x)')
    text = text.replace('\\operatorname{dan}', 'dan')
    text = text.replace('\\operatorname{Rp}', 'Rp')
    text = text.replace('\\overrightarrow{x}', 'x')
    text = text.replace('\\overrightarrow{PQ}', 'PQ')
    text = text.replace('\\overrightarrow{i}', 'j')
    text = text.replace('\\overrightarrow{j}', 'k')
    text = text.replace('\\vec', 'vec')
    text = text.replace('\\hat', 'hat')
    text = text.replace('\\text', '')
    text = text.replace('\\left', '')
    text = text.replace('\\right', '')
    text = text.replace('\\bar', '')
    text = text.replace('\\overline', '')
    text = text.replace('\\quad', '')
    text = text.replace('\\mathbb', '')
    text = text.replace('\\mathbf', '')
    text = text.replace('\\boldsymbol', '')
    text = text.replace('\\rightarrow', '->')
    text = text.replace('\\lim', 'lim ')
    text = text.replace('\\(y^{\prime}', 'y` ')
    text = text.replace('\\int_', 'integral dari')
    text = text.replace('\\times', 'x')
    text = text.replace('\\cong', 'kongruen')
    text = text.replace('\\equiv', 'ekuivalen')
    text = text.replace('\\pm', '+-')
    text = text.replace('\\neq', '=/=')
    text = text.replace('\\leq', '<=')
    text = text.replace('\\geq', '>=')
    text = text.replace('\\infty', 'tak hingga')
    text = text.replace('\\perp', 'tegak lurus')
    text = text.replace('\\bullet', '.')
    text = text.replace('\\imath', 'i')
    text = text.replace('\\mid', '|')
    text = text.replace('\\x |', '{x |')
    return text

def extract_t(i):
    try:
        return replacement_function(i['text'])
    except:
        return None

def extract_d(i):
    try:
        return i['data']
    except:
        return None

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

st.sidebar.title("Text Extraction App")
st.sidebar.write("")
st.sidebar.write("OCR text extraction from single or bulk images")

st.title("Streamlit_mathpix_OCR")
st.write("")
st.header("Bulk image processing")
st.text("Upload a csv file to extract text")
st.markdown('File must contain a column named **_imageUrl_**')

input_csv = st.file_uploader("Choose File", type="csv", accept_multiple_files=False, key=None, help=None)

if input_csv != None:
    image_data = pd.read_csv(input_csv)
    #image_data = image_data[0:2]
    files = image_data['imageUrl'].values
    result = []

    if st.button('Process File'):
        # Connecting to S3
        # Reading file_id file
        file_name = 'processed_file_id.csv'
        file = prefix + file_name
        obj = client.get_object(Bucket=bucket, Key=file)
        previous_file = pd.read_csv(obj['Body'])  # 'Body' is a key word

        all_file_ids = list(previous_file['file_id'].values)
        last_file_id = all_file_ids[-1]
        curr_id = last_file_id+1

        log_filename = f"process_log_{curr_id}.txt"

        logger = logging.getLogger("__process__") # Creating logging variable
        logger.setLevel(logging.DEBUG) # Set the minimun level of loggin to DEBUG
        
        formatter = logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s") # Logging format that will appear in the log file
        
        tail = TailLogger(10)

        log_handler = tail.log_handler #variable log handler
        log_handler.setFormatter(formatter) #set formatter to handler
        
        logger.addHandler(log_handler) # Adding file handler to logger

        logger.info("Bulk image processing start")
        with st.spinner('Extracting Text from Image ...'):
        
        # Using multiprocessing pool;
            start_time = datetime.datetime.now()
            with Pool(processes=128) as pool:
                try:
                    for i in stqdm(pool.imap(mathpix_text_asciimath_textapi, files), total=len(files)):
                        result.append(i)
                
                except Exception as ex:
                    logger.error(f"An exception of type {type(ex).__name__} occurred. Arguments: {ex.args}")

            time_taken = datetime.datetime.now() - start_time
            logger.info("Bulk image processing finished")
            text_df = image_data
            text_df['extracted_text'] = [extract_t(i) for i in result]
            text_df['extracted_equation'] = [extract_d(i) for i in result]

        st.success('Done!')
        st.write("Total time taken : " + str(time_taken))
        st.text(f"Processed Files : {len(text_df)}")
        st.text(f"Successful Files : {len(text_df[text_df['extracted_text'] != 'error'])}")
        st.text(f"Error Files : {len(text_df[text_df['extracted_text'] == 'error'])}")

        st.write("")
        st.write("current_file_id : " + str(curr_id))
        all_file_ids.append(curr_id)
        current_file = pd.DataFrame(all_file_ids, columns=['file_id'])

        #logging
        logger.info("Total time taken : " + str(time_taken))
        logger.info(f"Processed Files : {len(text_df)}")
        logger.info(f"Successful Files : {len(text_df[text_df['extracted_text'] != 'error'])}")
        logger.info(f"Error Files : {len(text_df[text_df['extracted_text'] == 'error'])}")
        logger.info(f"Files ID : {curr_id}")

        val_log = tail.contents()

        log_handler.close()
        logging.shutdown()
        logger.removeHandler(log_handler)
        del logger, log_handler
        
        # Writing updated file_id and new processed file
        output_file_names = ['sourcing_mathpix_'+str(curr_id)+'.csv', 'processed_file_id.csv']
        output_files = [text_df, current_file]

        for i in range(2):
            try:
                with StringIO() as csv_buffer:
                    output_files[i].to_csv(csv_buffer, index=False)
                    output_file = prefix + output_file_names[i]
                    print(output_file)
                    response = client.put_object(Bucket=bucket, Key=output_file, Body=csv_buffer.getvalue())
            except Exception as e:
                print(e)
        
        # Saving the log file to S3
        try:
            client.put_object(Bucket=bucket, Key=prefix + log_filename, Body=val_log)
            print(prefix + log_filename)
            print(val_log)
        except Exception as e:
            print(e)


#single image section
st.write("")
st.write("")
st.header("Single image processing")
st.text("Enter image URL to extract text")

input_url = st.text_input("image_URL")

if len(input_url) != 0:
    im = Image.open(requests.get(input_url, stream=True).raw)
    st.image(im)

    output_math = mathpix_text_asciimath_textapi(input_url)
    st.text(f"Input URL : {input_url}")
    st.text(f'Extracted Text : {output_math["text"]}')
    st.text(f'Extracted Equation : {output_math["data"][0]["value"]}')


# Download Section
st.write("")
st.write("")
st.header("Download processed file")
st.text("Type in file id to download processed file")
dwn_file_id = st.text_input("file_ID")

if dwn_file_id != "":
    dwn_file_name = 'sourcing_mathpix_'+str(dwn_file_id)+'.csv'
    dwn_file = prefix + dwn_file_name

    obj = client.get_object(Bucket= bucket, Key= dwn_file)
    dwn_data = pd.read_csv(obj['Body']) # 'Body' is a key word

    csv = convert_df(dwn_data)

    st.download_button(
                label="Download Result",
                data=csv,
                file_name='result'+str(dwn_file_id)+'.csv',
                mime='text/csv',
            )
