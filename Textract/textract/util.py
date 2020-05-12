import io
import json
import os
import re
import sys
import tarfile
import time
import webbrowser
from io import BytesIO
from pprint import pprint

import boto3
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def s3_to_image(bucket, document):
    # Get the document from S3
    s3 = boto3.resource('s3')
    s3_object = s3.Object(bucket, document)
    s3_response = s3_object.get()

    stream = io.BytesIO(s3_response['Body'].read())
    ori_image = Image.open(stream)
    image = Image.open(stream)
    return ori_image, image, stream


def get_sync_detect_document_text(bucket, document, response):

    ori_image, image, stream = s3_to_image(bucket, document)
    # Detect text in the document
    # process using image bytes
    # image_binary = stream.getvalue()
    # response = textract.detect_document_text(Document={'Bytes': image_binary})

    # Get the text blocks
    blocks = response['Blocks']
    width, height = image.size
    draw = ImageDraw.Draw(image)

    print('Detected Document Text')

    # Create image showing bounding box/polygon the detected lines/text
    for block in blocks:
        print('Type: ' + block['BlockType'])
        if block['BlockType'] != 'PAGE':
            print('Detected: ' + block['Text'])
            print('Confidence: ' + "{:.2f}".format(block['Confidence']) + "%")

        print('Id: {}'.format(block['Id']))
        if 'Relationships' in block:
            print('Relationships: {}'.format(block['Relationships']))
        print('Bounding Box: {}'.format(block['Geometry']['BoundingBox']))
        print('Polygon: {}'.format(block['Geometry']['Polygon']))
        print()
        draw = ImageDraw.Draw(image)
        # Draw WORD - Green -  start of word, red - end of word
        if block['BlockType'] == "WORD":
            draw.line([(width * block['Geometry']['Polygon'][0]['X'],
                        height * block['Geometry']['Polygon'][0]['Y']),
                       (width * block['Geometry']['Polygon'][3]['X'],
                        height * block['Geometry']['Polygon'][3]['Y'])], fill='green',
                      width=2)

            draw.line([(width * block['Geometry']['Polygon'][1]['X'],
                        height * block['Geometry']['Polygon'][1]['Y']),
                       (width * block['Geometry']['Polygon'][2]['X'],
                        height * block['Geometry']['Polygon'][2]['Y'])],
                      fill='red',
                      width=2)

        # Draw box around entire LINE
        if block['BlockType'] == "LINE":
            points = []

            for polygon in block['Geometry']['Polygon']:
                points.append((width * polygon['X'], height * polygon['Y']))

            draw.polygon((points), outline='black')

            # Uncomment to draw bounding box
            # box=block['Geometry']['BoundingBox']
            # left = width * box['Left']
            # top = height * box['Top']
            # draw.rectangle([left,top, left + (width * box['Width']), top +(height * box['Height'])],outline='black')
    return ori_image, image, blocks

# Displays information about a block returned by text detection and text analysis


def DisplayBlockInformation(block):
    print('Id: {}'.format(block['Id']))
    if 'Text' in block:
        print('    Detected: ' + block['Text'])
    print('    Type: ' + block['BlockType'])

    if 'Confidence' in block:
        print('    Confidence: ' + "{:.2f}".format(block['Confidence']) + "%")

    if block['BlockType'] == 'CELL':
        print("    Cell information")
        print("        Column:" + str(block['ColumnIndex']))
        print("        Row:" + str(block['RowIndex']))
        print("        Column Span:" + str(block['ColumnSpan']))
        print("        RowSpan:" + str(block['ColumnSpan']))

    if 'Relationships' in block:
        print('    Relationships: {}'.format(block['Relationships']))
    print('    Geometry: ')
    print('        Bounding Box: {}'.format(block['Geometry']['BoundingBox']))
    print('        Polygon: {}'.format(block['Geometry']['Polygon']))

    if block['BlockType'] == "KEY_VALUE_SET":
        print('    Entity Type: ' + block['EntityTypes'][0])
    if 'Page' in block:
        print('Page: ' + block['Page'])
    print()


def drawrect(drawcontext, xy, outline=None, width=1):
    (x1, y1), (x2, y2) = xy
    offset = 1
    for i in range(0, width):
        drawcontext.rectangle(((x1, y1), (x2, y2)), outline=outline)
        x1 = x1 - offset
        y1 = y1 + offset
        x2 = x2 + offset
        y2 = y2 - offset


def ShowBoundingBox(draw, box, width, height, boxColor):
    left = width * box['Left']
    top = height * box['Top']
    drawrect(draw, [(left, top), (left + (width * box['Width']),
                                  top + (height * box['Height']))], outline=boxColor, width=3)


def ShowSelectedElement(draw, box, width, height, boxColor):
    left = width * box['Left']
    top = height * box['Top']
    drawrect(draw, [(left, top), (left + (width * box['Width']),
                                  top + (height * box['Height']))], outline=boxColor, width=3)

# Displays information about a block returned by text detection and text analysis


def get_pdf_detect_document_text(document, blocks):

    from io import BytesIO
    from pdf2image import convert_from_path, convert_from_bytes

    image = convert_from_path(document)[0]
    ori_image = convert_from_path(document)[0]
    # Get the text blocks

    width, height = image.size
    draw = ImageDraw.Draw(image)

    print('Detected Document Text')

    # Create image showing bounding box/polygon the detected lines/text
    for block in blocks:
        print('Type: ' + block['BlockType'])
        if block['BlockType'] != 'PAGE':
            print('Detected: ' + block['Text'])
            print('Confidence: ' + "{:.2f}".format(block['Confidence']) + "%")

        print('Id: {}'.format(block['Id']))
        if 'Relationships' in block:
            print('Relationships: {}'.format(block['Relationships']))
        print('Bounding Box: {}'.format(block['Geometry']['BoundingBox']))
        print('Polygon: {}'.format(block['Geometry']['Polygon']))
        print()
        draw = ImageDraw.Draw(image)
        # Draw WORD - Green -  start of word, red - end of word
        if block['BlockType'] == "WORD":
            draw.line([(width * block['Geometry']['Polygon'][0]['X'],
                        height * block['Geometry']['Polygon'][0]['Y']),
                       (width * block['Geometry']['Polygon'][3]['X'],
                        height * block['Geometry']['Polygon'][3]['Y'])], fill='green',
                      width=2)

            draw.line([(width * block['Geometry']['Polygon'][1]['X'],
                        height * block['Geometry']['Polygon'][1]['Y']),
                       (width * block['Geometry']['Polygon'][2]['X'],
                        height * block['Geometry']['Polygon'][2]['Y'])],
                      fill='red',
                      width=2)

        # Draw box around entire LINE
        if block['BlockType'] == "LINE":
            points = []

            for polygon in block['Geometry']['Polygon']:
                points.append((width * polygon['X'], height * polygon['Y']))

            draw.polygon((points), outline='black')

            # Uncomment to draw bounding box
            # box=block['Geometry']['BoundingBox']
            # left = width * box['Left']
            # top = height * box['Top']
            # draw.rectangle([left,top, left + (width * box['Width']), top +(height * box['Height'])],outline='black')
    return ori_image, image, blocks


def get_sync_analyze_document(image, response):

    # Get the text blocks
    blocks = response['Blocks']
    width, height = image.size
    draw = ImageDraw.Draw(image)
    print('Detected Document Text')

    # Create image showing bounding box/polygon the detected lines/text
    for block in blocks:

        AnaylsisDisplayBlockInformation(block)

        draw = ImageDraw.Draw(image)
        if block['BlockType'] == "KEY_VALUE_SET":
            if block['EntityTypes'][0] == "KEY":
                ShowBoundingBox(
                    draw, block['Geometry']['BoundingBox'], width, height, 'red')
            else:
                ShowBoundingBox(
                    draw, block['Geometry']['BoundingBox'], width, height, 'green')

        if block['BlockType'] == 'TABLE':
            ShowBoundingBox(draw, block['Geometry']
                            ['BoundingBox'], width, height, 'blue')

        if block['BlockType'] == 'CELL':
            ShowBoundingBox(draw, block['Geometry']
                            ['BoundingBox'], width, height, 'yellow')
        if block['BlockType'] == 'SELECTION_ELEMENT':
            if block['SelectionStatus'] == 'SELECTED':
                ShowSelectedElement(
                    draw, block['Geometry']['BoundingBox'], width, height, 'blue')

            # uncomment to draw polygon for all Blocks
            # points=[]
            # for polygon in block['Geometry']['Polygon']:
            #    points.append((width * polygon['X'], height * polygon['Y']))
            # draw.polygon((points), outline='blue')
    return image, blocks


# Displays information about a block returned by text detection and text analysis

def get_page(blocks_list):
    page = ""
    blocks = [x for x in blocks_list if x['BlockType'] == "LINE"]
#     blocks.extend([x for x in blocks_list if x['BlockType'] == "WORD"])
    for block in blocks:
        page += " " + block['Text']
    print(page)
    return page


def detect_entities_for_comprehend(transcript):
    comprehend = boto3.client('comprehend')
    s3 = boto3.resource('s3')

    # print('Calling DetectDominantLanguage')
    res = comprehend.detect_dominant_language(Text=transcript)
    result = res['Languages'][0]
    LanguageCode = result['LanguageCode']
    # print("Language : {}, Score : {} \n\n".format(LanguageCode, result['Score']))

    # print("[detail result] \n" + json.dumps(res, sort_keys=True, indent=4))

    # print("End of DetectDominantLanguage\n")

    print('Calling DetectEntities')
    res = comprehend.detect_entities(
        Text=transcript, LanguageCode=LanguageCode)
    list_result = []
    for result in res['Entities']:
        list_result.append([result['Text'], result['Type'],
                            result['Score'], result['BeginOffset'], result['EndOffset']])
    df = pd.DataFrame(list_result, columns=[
                      'Text', 'Type', 'Score', 'BeginOffset', 'EndOffset'])
    df = df.sort_values(by='Score', ascending=False)
    return df


def get_kv_map(blocks):

    #     with open(file_name, 'rb') as file:
    #         img_test = file.read()
    #         bytes_test = bytearray(img_test)
    #         print('Image loaded', file_name)

    #     # process using image bytes
    #     client = boto3.client('textract')
    #     response = client.analyze_document(
    #         Document={'Bytes': bytes_test}, FeatureTypes=['FORMS'])

    #     # Get the text blocks
    #     blocks = response['Blocks']

    # get key and value maps
    key_map = {}
    value_map = {}
    block_map = {}
    for block in blocks:
        block_id = block['Id']
        block_map[block_id] = block
        if block['BlockType'] == "KEY_VALUE_SET":
            if 'KEY' in block['EntityTypes']:
                key_map[block_id] = block
            else:
                value_map[block_id] = block

    return key_map, value_map, block_map


def get_kv_relationship(key_map, value_map, block_map):
    kvs = {}
    for block_id, key_block in key_map.items():
        value_block = find_value_block(key_block, value_map)
        key = get_text(key_block, block_map)
        val = get_text(value_block, block_map)
        kvs[key] = val
    return kvs


def find_value_block(key_block, value_map):
    for relationship in key_block['Relationships']:
        if relationship['Type'] == 'VALUE':
            for value_id in relationship['Ids']:
                value_block = value_map[value_id]
    return value_block


def get_text(result, blocks_map):
    text = ''
    if 'Relationships' in result:
        for relationship in result['Relationships']:
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    word = blocks_map[child_id]
                    if word['BlockType'] == 'WORD':
                        text += word['Text'] + ' '
                    if word['BlockType'] == 'SELECTION_ELEMENT':
                        if word['SelectionStatus'] == 'SELECTED':
                            text += 'X '

    return text


def print_kvs(kvs):
    for key, value in kvs.items():
        print(key, ":", value)


def search_value(kvs, search_key):
    for key, value in kvs.items():
        if re.search(search_key, key, re.IGNORECASE):
            return value


def get_rows_columns_map(table_result, blocks_map):
    rows = {}
    for relationship in table_result['Relationships']:
        if relationship['Type'] == 'CHILD':
            for child_id in relationship['Ids']:
                cell = blocks_map[child_id]
                if cell['BlockType'] == 'CELL':
                    row_index = cell['RowIndex']
                    col_index = cell['ColumnIndex']
                    if row_index not in rows:
                        # create new row
                        rows[row_index] = {}

                    # get the text value
                    rows[row_index][col_index] = get_text(cell, blocks_map)
    return rows


def get_table_csv_results(blocks):

    blocks_map = {}
    table_blocks = []
    for block in blocks:
        blocks_map[block['Id']] = block
        if block['BlockType'] == "TABLE":
            table_blocks.append(block)

    if len(table_blocks) <= 0:
        return "<b> NO Table FOUND </b>"

    csv = ''
    for index, table in enumerate(table_blocks):
        csv += generate_table_csv(table, blocks_map, index + 1)
        csv += '\n\n'

    return csv


def generate_table_csv(table_result, blocks_map, table_index):
    rows = get_rows_columns_map(table_result, blocks_map)

    table_id = 'Table_' + str(table_index)

    # get cells.
    csv = 'Table: {0}\n\n'.format(table_id)

    for row_index, cols in rows.items():

        for col_index, text in cols.items():
            csv += '{}'.format(text) + ","
        csv += '\n'

    csv += '\n\n\n'
    return csv


# Display information about a block
def DisplayBlockInfo(block):

    print("Block Id: " + block['Id'])
    print("Type: " + block['BlockType'])
    if 'EntityTypes' in block:
        print('EntityTypes: {}'.format(block['EntityTypes']))

    if 'Text' in block:
        print("Text: " + block['Text'])

    if block['BlockType'] != 'PAGE':
        print("Confidence: " + "{:.2f}".format(block['Confidence']) + "%")

    print('Page: {}'.format(block['Page']))

    if block['BlockType'] == 'CELL':
        print('Cell Information')
        print('\tColumn: {} '.format(block['ColumnIndex']))
        print('\tRow: {}'.format(block['RowIndex']))
        print('\tColumn span: {} '.format(block['ColumnSpan']))
        print('\tRow span: {}'.format(block['RowSpan']))

        if 'Relationships' in block:
            print('\tRelationships: {}'.format(block['Relationships']))

    print('Geometry')
    print('\tBounding Box: {}'.format(block['Geometry']['BoundingBox']))
    print('\tPolygon: {}'.format(block['Geometry']['Polygon']))

    if block['BlockType'] == 'SELECTION_ELEMENT':
        print('    Selection element detected: ', end='')
        if block['SelectionStatus'] == 'SELECTED':
            print('Selected')
        else:
            print('Not selected')


def GetResults(jobId, types):
    maxResults = 1000
    paginationToken = None
    finished = False
    textract = boto3.client('textract')

    while finished == False:

        response = None

        if types == 2:
            if paginationToken == None:
                response = textract.get_document_analysis(JobId=jobId,
                                                          MaxResults=maxResults)
            else:
                response = textract.get_document_analysis(JobId=jobId,
                                                          MaxResults=maxResults,
                                                          NextToken=paginationToken)

        if types == 1:
            if paginationToken == None:
                response = textract.get_document_text_detection(JobId=jobId,
                                                                MaxResults=maxResults)
            else:
                response = textract.get_document_text_detection(JobId=jobId,
                                                                MaxResults=maxResults,
                                                                NextToken=paginationToken)

        blocks = response['Blocks']
        print('Detected Document Text')
        print('Pages: {}'.format(response['DocumentMetadata']['Pages']))

        # Display block information
        for block in blocks:
            DisplayBlockInfo(block)
            print()
            print()

        if 'NextToken' in response:
            paginationToken = response['NextToken']
        else:
            finished = True
    return blocks


def GetResultsDocumentAnalysis(jobId):
    maxResults = 1000
    paginationToken = None
    finished = False
    textract = boto3.client('textract')

    while finished == False:

        response = None
        if paginationToken == None:
            response = textract.get_document_analysis(JobId=jobId,
                                                      MaxResults=maxResults)
        else:
            response = textract.get_document_analysis(JobId=jobId,
                                                      MaxResults=maxResults,
                                                      NextToken=paginationToken)

        # Get the text blocks
        blocks = response['Blocks']
        print('Analyzed Document Text')
        print('Pages: {}'.format(response['DocumentMetadata']['Pages']))
        # Display block information
        for block in blocks:
            DisplayBlockInfo(block)
            print()
            print()

            if 'NextToken' in response:
                paginationToken = response['NextToken']
            else:
                finished = True
    return blocks


def get_pdf_analyze_document(document, blocks):

    from io import BytesIO
    from pdf2image import convert_from_path, convert_from_bytes

    image = convert_from_path(document)[0]
    ori_image = convert_from_path(document)[0]

    width, height = image.size
    draw = ImageDraw.Draw(image)
    print('Detected Document Text')

    # Create image showing bounding box/polygon the detected lines/text
    for block in blocks:

        AnaylsisDisplayBlockInformation(block)

        draw = ImageDraw.Draw(image)
        if block['BlockType'] == "KEY_VALUE_SET":
            if block['EntityTypes'][0] == "KEY":
                ShowBoundingBox(
                    draw, block['Geometry']['BoundingBox'], width, height, 'red')
            else:
                ShowBoundingBox(
                    draw, block['Geometry']['BoundingBox'], width, height, 'green')

        if block['BlockType'] == 'TABLE':
            ShowBoundingBox(draw, block['Geometry']
                            ['BoundingBox'], width, height, 'blue')

        if block['BlockType'] == 'CELL':
            ShowBoundingBox(draw, block['Geometry']
                            ['BoundingBox'], width, height, 'yellow')
        if block['BlockType'] == 'SELECTION_ELEMENT':
            if block['SelectionStatus'] == 'SELECTED':
                ShowSelectedElement(
                    draw, block['Geometry']['BoundingBox'], width, height, 'blue')

            # uncomment to draw polygon for all Blocks
            # points=[]
            # for polygon in block['Geometry']['Polygon']:
            #    points.append((width * polygon['X'], height * polygon['Y']))
            # draw.polygon((points), outline='blue')
    return ori_image, image, blocks


def AnaylsisDisplayBlockInformation(block):
    print('Id: {}'.format(block['Id']))
    if 'Text' in block:
        print('    Detected: ' + block['Text'])
    print('    Type: ' + block['BlockType'])

    if 'Confidence' in block:
        print('    Confidence: ' + "{:.2f}".format(block['Confidence']) + "%")

    if block['BlockType'] == 'CELL':
        print("    Cell information")
        print("        Column:" + str(block['ColumnIndex']))
        print("        Row:" + str(block['RowIndex']))
        print("        Column Span:" + str(block['ColumnSpan']))
        print("        RowSpan:" + str(block['ColumnSpan']))

    if 'Relationships' in block:
        print('    Relationships: {}'.format(block['Relationships']))
    print('    Geometry: ')
    print('        Bounding Box: {}'.format(block['Geometry']['BoundingBox']))
    print('        Polygon: {}'.format(block['Geometry']['Polygon']))

    if block['BlockType'] == "KEY_VALUE_SET":
        print('    Entity Type: ' + block['EntityTypes'][0])

    if block['BlockType'] == 'SELECTION_ELEMENT':
        print('    Selection element detected: ', end='')

        if block['SelectionStatus'] == 'SELECTED':
            print('Selected')
        else:
            print('Not selected')

    if 'Page' in block:
        print('Page: ' + str(block['Page']))
    print()


def get_multipages_block_result(jobId):
    maxResults = 1000
    paginationToken = None
    finished = False
    textract = boto3.client('textract')

    doc_block = {}
    init_val = False
    while finished == False:
        #         print("paginationToken : {}".format(paginationToken))
        response = None
        if paginationToken == None:
            response = textract.get_document_analysis(JobId=jobId,
                                                      MaxResults=maxResults)
        else:
            response = textract.get_document_analysis(JobId=jobId,
                                                      MaxResults=maxResults,
                                                      NextToken=paginationToken)

        # Get the text blocks
        blocks = response['Blocks']

        if not init_val:
            print('Analyzed Document Text')
            print('Pages: {}'.format(response['DocumentMetadata']['Pages']))
            for i in range(int(response['DocumentMetadata']['Pages'])):
                doc_block[i] = []
                init_val = True

        for block in blocks:
            doc_block[int(block['Page'])-1].append(block)

            if 'NextToken' in response:
                paginationToken = response['NextToken']
            else:
                finished = True
    return doc_block


def get_multipages_analyze_document(image, blocks):

    width, height = image.size
    draw = ImageDraw.Draw(image)
    print('Detected Document Text')

    # Create image showing bounding box/polygon the detected lines/text
    for block in blocks:

        AnaylsisDisplayBlockInformation(block)

        draw = ImageDraw.Draw(image)
        if block['BlockType'] == "KEY_VALUE_SET":
            if block['EntityTypes'][0] == "KEY":
                ShowBoundingBox(
                    draw, block['Geometry']['BoundingBox'], width, height, 'red')
            else:
                ShowBoundingBox(
                    draw, block['Geometry']['BoundingBox'], width, height, 'green')

        if block['BlockType'] == 'TABLE':
            ShowBoundingBox(draw, block['Geometry']
                            ['BoundingBox'], width, height, 'blue')

        if block['BlockType'] == 'CELL':
            ShowBoundingBox(draw, block['Geometry']
                            ['BoundingBox'], width, height, 'yellow')
        if block['BlockType'] == 'SELECTION_ELEMENT':
            if block['SelectionStatus'] == 'SELECTED':
                ShowSelectedElement(
                    draw, block['Geometry']['BoundingBox'], width, height, 'blue')

            # uncomment to draw polygon for all Blocks
            # points=[]
            # for polygon in block['Geometry']['Polygon']:
            #    points.append((width * polygon['X'], height * polygon['Y']))
            # draw.polygon((points), outline='blue')
    return image, blocks
