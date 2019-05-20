def getImage(imgFile, imageNum):
  imgF = open(imgFile, "rb")     # and pixel vals separate
  output = []

  imgF.read(16 + 784*imageNum)   # discard header info and all unwanted previous images

  for j in range(784):  # get 784 vals from the image file
     val = ord(imgF.read(1)) #get pixel data
     output.append(val) #add data to output

  imgF.close()
  return output

def getLabel(lblFile, imageNum):
  lblF = open(lblFile)
  lblF.read(8 + imageNum)

  out = ord(lblF.read(1))

  lblF.close();
  return(out)
