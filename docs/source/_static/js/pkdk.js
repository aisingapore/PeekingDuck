
// Script for Training Pipeline Pages
$(window).on("load", function (e) {
    var framework_btn_py = $('.training-framework-py');
    var content_py = $('.training-pytorch');
    var framework_btn_tf = $('.training-framework-tf');
    var content_tf = $('.training-tensorflow');
    
    content_py.show();

    framework_btn_py.on('click', function(){ 
        framework_btn_tf.removeClass("active");
        framework_btn_py.addClass("active");
        content_py.show();
        content_tf.hide();
    } );
    
    framework_btn_tf.on('click', function(){ 
        framework_btn_py.removeClass("active");
        framework_btn_tf.addClass("active");
        content_tf.show();
        content_py.hide();
    } );
  })