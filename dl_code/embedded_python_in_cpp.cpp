#include <iostream>
#include <Python.h>

int main() {
	//Initialize the python instance
	Py_Initialize();


  // case 1 : 	
	//Run a simple string
	//PyRun_SimpleString("from time import time,ctime\n"
  //						"print('Today is',ctime(time()))\n");

  // case 2 :
	//Run a simple file
	FILE* PScriptFile = fopen("test.py", "r");
	if(PScriptFile){
		PyRun_SimpleFile(PScriptFile, "test.py");
		fclose(PScriptFile);
	}

  // case 3 :
	////Run a python function
	//PyObject *pName, *pModule, *pFunc, *pArgs, *pValue;

	//pName = PyUnicode_FromString((char*)"script");
	//pModule = PyImport_Import(pName);
	//pFunc = PyObject_GetAttrString(pModule, (char*)"test");
	//pArgs = PyTuple_Pack(1, PyUnicode_FromString((char*)"Greg"));
	//pValue = PyObject_CallObject(pFunc, pArgs);
	//
	//auto result = _PyUnicode_AsString(pValue);
	//std::cout << result << std::endl;



	//Close the python instance
	Py_Finalize();

  return 0;
}
