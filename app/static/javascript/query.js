// Set the default value to 0 when the input box is empty and remove it when values are entered
function formatInputValue(input) {
    if (input.value === "") {
        input.value = "0";
    } else if (input.value === "0") {
        input.value = "";
    } else {
        input.value = parseInt(input.value).toString();
    }
}

document.addEventListener('DOMContentLoaded', function() {
    var form = document.querySelector('form[action="/prediction"]');
    var inputs = form.querySelectorAll('input[type="number"]');

    inputs.forEach(function(input) {
      input.addEventListener('input', function() {
        if (this.value === '') {
          this.value = '0';
        }
      });

      input.addEventListener('keydown', function() {
        if (this.value === '0') {
          this.value = '';
        }
      });

      input.addEventListener('blur', function() {
        if (this.value === '') {
          this.value = '0';
        }
        this.value = parseInt(this.value);
      });
    });
  });

// Reset the input box values to 0 when the page is reloaded
window.addEventListener('DOMContentLoaded', function () {
    var inputs = document.getElementsByTagName('input');
    for (var i = 0; i < inputs.length; i++) {
        inputs[i].value = "0";
    }
});

