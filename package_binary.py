import sys
import subprocess
import shutil
print sys.argv[1]

result = subprocess.check_output(["./setup.sh ldd ./run_align"], shell=True)


for l in result.split("\n"):
  try:
    info = l.split("=>")[1].strip()
    #info = info.substr(info.find("("), info.find(")")-info.find("("))
    info = info.split(" ")[0].strip()
    shutil.copyfile(info, "package/"+info.split("/")[-1])
  except:
    print "Exception"
    pass
