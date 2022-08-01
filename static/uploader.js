$(document).ready(function(){
    $('form input').change(function () {
      $('form p').text(this.files.length + " file(s) selected");
    });
  });



function modalerror(f){
  if(f != 0){
        swal({
      title: "Attenzione!",
      text: "Nella foto caricata non è stata rilevata nessuna faccia.",
      icon: "error",
      button: "Ok!",
    })
  }
}
