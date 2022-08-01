
function update(){
    currentfps = document.getElementById('fpslidershow').textContent;
    json = {"fps": currentfps}
    pass = JSON.stringify(json)
        $.ajax({
            type: 'POST',
            url: '/settings-RU',
            data: pass,
            contentType: "application/json, charset=utf-8",
            success: function (data) {
                swal({
                    title: "Fatto!",
                    text: "Le verifiche sono state salvate con successo!",
                    icon: "success",
                    button: "Ok!",
                  });
            },
            
    });
}


