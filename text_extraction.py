import streamlit as st
from PIL import Image
import requests
import pandas as pd
import datetime
from stqdm import stqdm
import boto3
from io import StringIO
import logging
import collections
import asyncio
import aiohttp
from math import ceil
from time import sleep
import re

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
      
async def mathpix_text_asciimath_textapi(logger, session, url):
    tries = 3
    global error_amount

    data = {
        "src": url,
        "formats": ["data", "text"],
        "data_options": {"include_asciimath": True},
        "include_smiles": True
    }
    
    for i in range(tries):
        try:
            r = session.post("https://api.mathpix.com/v3/text",
                              json=data,
                              headers={
                                  "app_id": app_id,
                                  "app_key": app_key,
                                  "Content-type": "application/json"
                              })
            response = await r
            return response
        except Exception as e:
            if i < (tries - 1):
                continue
            else:
                print(e)
                logger.error(f"Error Processing : {url} {e}") # error logging
                error_amount += 1

    return asyncio.sleep(1) # returning none if process error

# regular mathpix function without async
def regular_mathpix(url):
    data = {
        "src": url,
        "formats": ["data", "text"],
        "data_options": {"include_asciimath": True},
        "include_smiles": True
    }
    
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
        response = e

    return response
  
# Python Escape Character
# \\
# \'
# \"
# \a
# \b
# \f
# \n
# \r
# \t

regex_extract_list = [
                      r"\\mathrm[({]([a-zA-Z0-9~]*)[})]",
                      r"\\operatorname[({]([a-zA-Z0-9~]*)[})]"
]

regex_remove_list = [
                     r"\\begin{[a-zA-Z0-9|:]*}{[a-zA-Z0-9|:]*}",
                     r"\\end{[a-zA-Z0-9|:]*}"
]

def replacement_function(text):
    text = text.replace('\\mathrm{~cm}', 'cm')
    text = text.replace('\\mathrm{~jam}', 'jam')
    text = text.replace('\\mathrm{~Rp.}', 'Rp.')
    text = text.replace('\\mathrm{~Hz}', 'Hz')
    text = text.replace('\\mathrm{~dm}', 'dm')
    text = text.replace('\\mathrm{~kg}', 'kg')
    text = text.replace('\\mathrm{~m}', 'm')
    
    # extraction regex
    for i in regex_extract_list:
        try:
            text = re.sub(i, r"\1", text)
        except:
            continue
    
    # removing regex
    for i in regex_remove_list:
        try:
            text = re.sub(i, '', text)
        except:
            continue
    
    text = text.replace('\\mathrm{cm}', 'cm')
    text = text.replace('\\mathrm{jam}', 'jam')
    text = text.replace('\\mathrm{Rp.}', 'Rp.')
    text = text.replace('\\mathrm{Hz}', 'Hz')
    text = text.replace('\\mathrm{dm}', 'dm')
    text = text.replace('\\mathrm{kg}', 'kg')
    text = text.replace('\\mathrm{P}', 'P')
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
    text = text.replace("\\ldots", "...")
    text = text.replace("\\cdots", "..")
    text = text.replace("\\cdot", ".")
    text = text.replace(r'^{\circ}', '')
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
    text = text.replace('\text', '')
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
    text = text.replace(r'\(y^{\prime}', 'y` ')
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
    text = text.replace(r"\wedge", "^")
    text = text.replace(r"\vee", "v")
    text = text.replace(r"\sim", "~")
    text = text.replace(r"\Rightarrow", "->")
    text = text.replace(r"\Leftarrow", "<-")
    text = text.replace(r"\Leftrightarrow", "<=>")
    text = text.replace(r"\sum_", "sigma")
    text = text.replace(r"\sum", "sigma")
    text = text.replace(r"arrow", "->")
    text = text.replace(r"\ldots", "....")
    text = text.replace(r"\(", "")
    text = text.replace(r"\)", "")
    text = text.replace(r"\[", "")
    text = text.replace(r"\]", "")
    text = text.replace(r"(\begin{array}", "")
    text = text.replace(r"\end{array})", "")
    text = text.replace(r"\begin{tabular}", "")
    text = text.replace(r"\end{tabular}", "")
    text = text.replace(r"&", "")
    text = text.replace(r"^{\prime\prime}", "\'\'")
    text = text.replace(r"\hline", "")
    text = text.replace(r"\cline", "")
    text = text.replace(r"\multirow", "")
    text = text.replace(r"\multicolumn", "")
    text = text.replace(r"^{\prime}", "\'")
    text = text.replace(r"\div", "/")
    text = text.replace(r"vec{\imath}", "i")
    text = text.replace(r"vec{\jmath}", "j")
    text = text.replace(r"vec{\kmath}", "k")
    text = text.replace("vec{i}", "i")
    text = text.replace("vec{j}", "j")
    text = text.replace("vec{k}", "k")
    text = text.replace("hat{i}", "i")
    text = text.replace("hat{j}", "j")
    text = text.replace("hat{k}", "k")
    text = text.replace("vec{a}", "a")
    text = text.replace("vec{b}", "b")
    text = text.replace("vec{c}", "c")
    text = text.replace("vec{d}", "d")
    text = text.replace("vec{u}", "u")
    text = text.replace("vec{v}", "v")
    text = text.replace("vec{w}", "w")
    text = text.replace("\\frac{", "(")
    text = text.replace("}{", ")/(")
    text = text.replace("\\{", "kurungbukakurawal")
    text = text.replace("\\}", "kurungtutupkurawal")
    text = text.replace("}", ")")
    text = text.replace("_{", "_(")
    text = text.replace("kurungbukakurawal", "{")
    text = text.replace("kurungtutupkurawal","}")
    text = text.replace(r"\sqrt{", "akar(")
    text = text.replace("^{", "^(")
    text = text.replace(r"\rfloor", " ]")
    text = text.replace(r"\lfloor", "[ ")
    text = text.replace(r"\\", "")
    text = text.replace(r"\\\\", "")
    text = text.replace("\\", "")
    text = text.replace("\\\\", "")
    text = text.replace("~N", "N")
    text = text.replace("~km", "km")
    text = text.replace("~Pa", "Pa")
    text = text.replace("~J", "J")
    text = text.replace("~F", "F")
    text = text.replace("~s", "s")
    text = text.replace("~mm", "mm")
    text = text.replace("~W", "W")
    text = text.replace("~g", "g")
    text = text.replace("~A", "A")
    text = text.replace("~K", "K")
    text = text.replace("~dB", "dB")
    text = text.replace("~V", "V")
    text = text.replace("~mL", "mL")
    text = text.replace("~mol", "mol")
    text = text.replace("~T", "T")
    text = text.replace("~Wb", "Wb")

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

def get_task(logger, session, start_index, end_index):
    tasks = []
    
    for i in files[start_index:end_index]:
        tasks.append(mathpix_text_asciimath_textapi(logger, session, i))
    return tasks

async def math_calls(logger):
    async with aiohttp.ClientSession() as session:
        for i in stqdm(range(ceil(file_len / batch_count))):
            process_start = datetime.datetime.now()

            if (i+1)*batch_count < file_len:
                tasks = get_task(logger, session, i*batch_count, (i+1)*batch_count)
            else:
                tasks = get_task(logger, session, i*batch_count, file_len)

            responses = await asyncio.gather(*tasks)
            for response in responses:
                if type(response) == aiohttp.ClientResponse:
                    try:
                        result.append(await response.json())
                    except Exception as e:
                        result.append(None)
                        logger.error(f"Error Happen : {e}")
                else:
                    result.append(await response)
            
            process_taken = datetime.datetime.now() - process_start
            logger.debug(f"{i} - {process_taken}")
            if process_taken.seconds < batch_timing:
                sleep(batch_timing-process_taken.seconds)

@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')

def check_limit():
    limit_name = 'daily_limit.csv'
    limit_file = prefix + limit_name
    obj = client.get_object(Bucket=bucket, Key=limit_file)
    daily_process_df = pd.read_csv(obj['Body'])  # 'Body' is a key word

    try:
        #getting the current date if there is already any
        todays_proc_def = int(daily_process_df[daily_process_df['date'] == f'{str(datetime.date.today())}']['proc_qty'])
    except:
        #if not, create new date for the dataframe
        todays_proc_def = 0
        daily_process_df = pd.concat([daily_process_df, pd.DataFrame([{'date': f'{str(datetime.date.today())}', 'proc_qty': 0}])])
    
    return todays_proc_def, daily_process_df

todays_proc, daily_process = check_limit()
daily_limit_amount = 1000

# === SIDEBAR ===
st.sidebar.title("Text Extraction App")
st.sidebar.write("")
st.sidebar.write("OCR text extraction from single or bulk images")

# === TITLE ===
st.title("Streamlit Mathpix OCR")
st.write("")


# === BULK IMAGE PROCESSING SECTION ===
st.header("Bulk Image Processing")
col1, col2, col3 = st.columns([5,1,1])

with col1:
    st.text("Upload a csv file to extract text")
    st.markdown('File must contain a column named **_imageUrl_**')

with col2:
    st.metric("Processed", f"{todays_proc}")

with col3:
    limit_left = daily_limit_amount-todays_proc
    if limit_left < 0:
        limit_left = 0
    st.metric("Limit", limit_left)

input_csv = st.file_uploader("Choose File", type="csv", accept_multiple_files=False, key=None, help=None)

if input_csv != None:
    image_data = pd.read_csv(input_csv)
    files = image_data['imageUrl'].values
    file_len = len(files) #total amount of file in the csv
    batch_count = 100 # amount of API call send every async batch
    batch_timing = ceil(60/1000*batch_count)

    todays_proc, daily_process = check_limit()

    if daily_limit_amount-(file_len + todays_proc) < 0:
        st.error("Limit is reached, please try tomorrow")

    if (st.button('Process File')) and (daily_limit_amount-(file_len + todays_proc) >= 0):
        result = []
        error_amount = 0

        # Connecting to S3
        # Reading file_id file
        file_name = 'processed_file_id.csv'
        file = prefix + file_name
        obj = client.get_object(Bucket=bucket, Key=file)
        previous_file = pd.read_csv(obj['Body'])  # 'Body' is a key word

        all_file_ids = list(previous_file['file_id'].values)
        last_file_id = all_file_ids[-1]
        curr_id = last_file_id+1

        # logging initialization
        log_filename = f"process_log_{curr_id}.txt" # the name of the log file
        logger = logging.getLogger("__process__") # creating logging variable
        logger.setLevel(logging.DEBUG) # set the minimun level of loggin to DEBUG
        formatter = logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s") # logging format that will appear in the log file
        tail = TailLogger(10000) # amount of log that is saved
        log_handler = tail.log_handler # variable log handler
        log_handler.setFormatter(formatter) # set formatter to handler
        logger.addHandler(log_handler) # adding file handler to logger
        
        with st.spinner('Extracting Text from Image ...'):
            start_time = datetime.datetime.now()
            logger.info("Bulk image processing start")
            
            asyncio.run(math_calls(logger)) #starting async process

            time_taken = datetime.datetime.now() - start_time
            logger.info("Bulk image processing finished")

            # extracting the result
            text_df = image_data
            text_df['extracted_text'] = [extract_t(i) for i in result]
            text_df['extracted_equation'] = [extract_d(i) for i in result]

        # process info
        st.success('Done!')
        st.text("Total time taken : " + str(time_taken))
        st.write("")
        st.text(f"Processed Files : {len(text_df)}")
        st.text(f"Successful Files : {len(text_df) - error_amount}")
        st.text(f"Error Files : {error_amount}")
        st.write("")
        st.text(f"Text Result : {len(text_df[~text_df['extracted_text'].isna()])}")
        st.text(f"None Result : {len(text_df[text_df['extracted_text'].isna()])}")

        # file id for upload and download to s3
        st.write("")
        st.text("current_file_id : " + str(curr_id))
        all_file_ids.append(curr_id)
        current_file = pd.DataFrame(all_file_ids, columns=['file_id'])

        #logging
        logger.info("Total time taken : " + str(time_taken))
        logger.info(f"Processed Files : {len(text_df)}")
        logger.info(f"Successful Files : {len(text_df) - error_amount}")
        logger.info(f"Error Files : {error_amount}")
        logger.info(f"Text Result : {len(text_df[~text_df['extracted_text'].isna()])}")
        logger.info(f"None Result : {len(text_df[text_df['extracted_text'].isna()])}")
        logger.info(f"File ID : {curr_id}")

        val_log = tail.contents() # extracting the log 

        # deleting all loggin variable for the current process
        log_handler.close()
        logging.shutdown()
        logger.removeHandler(log_handler)
        del logger, log_handler

        todays_proc, daily_process = check_limit()
        daily_process.loc[daily_process['date'] == f'{str(datetime.date.today())}', 'proc_qty'] = todays_proc + file_len # updating the number of file in csv
        
        # writing updated file_id and new processed file
        output_file_names = ['sourcing_mathpix_'+str(curr_id)+'.csv', 'processed_file_id.csv', 'daily_limit.csv']
        output_files = [text_df, current_file, daily_process]

        for i in range(3):
            try:
                with StringIO() as csv_buffer:
                    output_files[i].to_csv(csv_buffer, index=False)
                    output_file = prefix + output_file_names[i]
                    print(output_file)
                    response = client.put_object(Bucket=bucket, Key=output_file, Body=csv_buffer.getvalue())
            except Exception as e:
                print(e)
        
        # saving the log file to S3
        try:
            client.put_object(Bucket=bucket, Key=prefix + log_filename, Body=val_log)
            print(prefix + log_filename)
            print(val_log)
        except Exception as e:
            print(e)


# === SINGLE IMAGE SECTION ===
st.write("")
st.write("")
st.header("Single Image Processing")
st.text("Enter image URL to extract text")

input_url = st.text_input("Image URL")

if len(input_url) != 0:
    try:
        im = Image.open(requests.get(input_url, stream=True).raw)
        st.image(im)
        output_math = regular_mathpix(input_url)
        try:
            st.text(f'Extracted Text : {output_math["text"]}')
            #st.text(f'Extracted Equation : {output_math["data"][0]["value"]}')
            cln_text = replacement_function(output_math['text'])
            st.text(f'Cleaned Output : {cln_text}')
        except:
            st.text(f'Extracted Equation : None')
    except:
        st.text("Error processing, the URL is not an image")


# === DOWNLOAD SECTION ===
st.write("")
st.write("")
st.header("Download Processed File")
st.text("Type in file id to download processed file")
dwn_file_id = st.text_input("File ID")

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
