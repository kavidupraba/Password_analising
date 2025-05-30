f=(8,[2,3,4,7],[0.4982966132373168,0.5004787247622318,0.0010222990256835995,0.00020236297476783566])
fc=f[0]
fi=f[1]
ff=f[2]

fe=[0]*fc
for i,val in enumerate(fi):
    fe[val]=ff[i]
print(fe)
check="0005AD76BD555C1D6D771DE417A4B87E4B4"
counter=0
for c in check:
    counter+=1
print(counter)

