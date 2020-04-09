//get the input and UL list
var input = document.getElementById('filesToUpload');
var list = document.getElementById('fileList');
var filecount;

//for every file...
for (var x = 0; x < input.files.length; x++) {
	//add to list
	var li = document.createElement('li');
	li.innerHTML = 'File ' + (x + 1) + ':  ' + input.files[x].name;
	list.append(li);
	filecount += 1;
}

function makeFileList() {
	var input = document.getElementById("filesToUpload");
	var ul = document.getElementById("fileList");
	while (ul.hasChildNodes()) {
		ul.removeChild(ul.firstChild);
	}
	for (var i = 0; i < input.files.length; i++) {
		var li = document.createElement("li");

		if ( 4 < i  ) {
				li.innerHTML = '...<br>' + input.files[input.files.length-1].name;
				ul.appendChild(li);
				break;
		}
		else {
			li.innerHTML = input.files[i].name;
			ul.appendChild(li);
		}

	}
	if(!ul.hasChildNodes()) {
		var li = document.createElement("li");
		li.innerHTML = 'No Files Selected';
		ul.appendChild(li);
	}
}