import gspread
from oauth2client.service_account import ServiceAccountCredentials

# https://github.com/burnash/gspread
# 操作 https://tanuhack.com/library-gspread/

scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']

credentials = ServiceAccountCredentials.from_json_keyfile_name('nlp2019-f28ff672b2db.json', scope)
gc = gspread.authorize(credentials)

### select workbook of file ###
workbook = gc.open('nlp2019').sheet1

### get sheet ###
wks_list = workbook.worksheets()
print("worksheet_0_title: {}\t sheet_id: {}".format(wks_list[0].title, wks_list[0].id))

### select sheet ###
worksheet = workbook.worksheet('sample') # sheet name
worksheet.update_title('sample_改')

### create/delete sheet ###
workboook.add_worksheet(title="hoge", rows=100, cols=26)
workbook.del_worksheet(worksheet)

### operate cell ###

# 1. ラベル指定
cell_value = worksheet.acell('A1').value
print(cell_value)

# 2. 行番号と列番号を指定
cell_value = worksheet.cell(1,2).value

# 3. ラベルを指定，複数セル値を一次配列に格納
range_list = worksheet.range("A1:B10")
row_list = worksheet.row_values(1) # 第二引数が2の場合，数式を格納
col_list = worksheet.col_values(1) #
all_list = worksheet.get_all_values()


### update cell ###

worksheet.update_acell('A1', 'Hello World!')
print(worksheet.acell('A1'))


### search ###
cell_list = worksheet.findall("World")
print("行番号:{}, 列番号:{}".format(cell_list[0].row, cell_list[0].col))
worksheet.update_cell(cell_list[0].row, cell_list[0].col, "Real World")
