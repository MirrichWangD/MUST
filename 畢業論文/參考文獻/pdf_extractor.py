import PyPDF2

# 打开原始 PDF 文件
input_pdf_path = 'PROCEEDINGOFTHEINTERNATIONALCONFERENCE_ICETSD19.pdf'
output_pdf_path = 'output.pdf'

# 要提取的页面索引（例如提取第1-3页，注意索引从0开始）
pages_to_extract = [507,508,509]  # 代表第一页到第三页

# 读取 PDF 文件
with open(input_pdf_path, 'rb') as pdf_file:
    reader = PyPDF2.PdfReader(pdf_file)
    writer = PyPDF2.PdfWriter()

    # 提取指定页面并添加到 writer 中
    for page_num in pages_to_extract:
        page = reader.pages[page_num]
        writer.add_page(page)

    # 将提取的页面保存到新的 PDF 文件
    with open(output_pdf_path, 'wb') as output_pdf:
        writer.write(output_pdf)

print("PDF 页面提取完成:", output_pdf_path)