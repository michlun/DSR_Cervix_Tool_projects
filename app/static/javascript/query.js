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

        // Store the input values
        function storeData() {
            var formData = new FormData();
            var inputs = document.getElementsByTagName('input');
            for (var i = 0; i < inputs.length; i++) {
                if (inputs[i].value == 0 || inputs[i].value == '') {
                    formData.append(inputs[i].id, 0);
                } else {
                    formData.append(inputs[i].id, inputs[i].value);
                }
        }

            var xhr = new XMLHttpRequest();
            xhr.open('POST', '/predictdata', true);
            xhr.onload = function () {
                if (xhr.status === 200) {
                    window.location.href = '/prediction.html';
                } else {
                    alert('An error is occurred');
                }
            };
            xhr.send(formData);
        }

        // Reset the input box values to 0 when the page is reloaded
        window.addEventListener('DOMContentLoaded', function () {
            var inputs = document.getElementsByTagName('input');
            for (var i = 0; i < inputs.length; i++) {
                inputs[i].value = "0";
            }
        });