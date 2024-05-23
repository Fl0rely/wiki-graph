import graph_function as gf
#open simple english 
link="doc\\dump_simple_wiki.xml"

#dump=open("C:\\Users\\virel\\Desktop\\prépa\\TIPE\\tiwiki-20240320-pages-articles-multistream.xml\\tiwiki-20240320-pages-articles-multistream.xml","r",encoding="utf-8")

res="res.txt"
lim_page=10000
count=gf.parse_lxml(link,res,lim_page)
gf.make_graph(res,lim_page)
