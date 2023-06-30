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

// For the model 2
function toggle_heatmap() {
        heatmap_on = !heatmap_on;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.globalAlpha = 1.0;
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        if (heatmap_on == true) {
          ctx.globalAlpha = 0.3;
          ctx.drawImage(heatmap, 0, 0, canvas.width, canvas.height);
        }
      };

      var scale = {{ img_scale }};
      var heatmap_on = false;
      var canvas = document.getElementById('image-canvas');
      var ctx = canvas.getContext('2d');
      var img = new Image();
      img.src = "data:image/jpeg;base64,{{ image }}";
      var heatmap = new Image();
      heatmap.src = "data:image/jpeg;base64,{{ heatmap }}";

      img.onload = function() {
        canvas.width = img.width / scale;
        canvas.height = img.height / scale;
        ctx.globalAlpha = 1.0;
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        if (heatmap_on == true) {
          ctx.globalAlpha = 0.3;
          ctx.drawImage(heatmap, 0, 0, canvas.width, canvas.height);
        }
      }

      var isDrawing = false;
      var startX, startY;

      canvas.addEventListener('mousedown', function(event) {
        isDrawing = true;
        startX = event.offsetX;
        startY = event.offsetY;
      });

      canvas.addEventListener('mousemove', function(event) {
        if (!isDrawing) return;

        var x = event.offsetX;
        var y = event.offsetY;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.globalAlpha = 1.0;
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        if (heatmap_on == true) {
          ctx.globalAlpha = 0.3;
          ctx.drawImage(heatmap, 0, 0, canvas.width, canvas.height);
        }
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.strokeRect(startX, startY, x - startX, y - startY);
      });

      canvas.addEventListener('mouseup', function(event) {
        isDrawing = false;

        var x = event.offsetX;
        var y = event.offsetY;

        var xstart = Math.min(startX, x) * scale;
        var xend = Math.max(startX, x) * scale;
        var ystart = Math.min(startY, y) * scale;
        var yend = Math.max(startY, y) * scale;

        document.getElementById('xstart').value = xstart;
        document.getElementById('xend').value = xend;
        document.getElementById('ystart').value = ystart;
        document.getElementById('yend').value = yend;

        // Uncomment the following line if you want to submit the form automatically after the crop
        // document.getElementById('annotate-form').submit();
      });
      button_heatmap.addEventListener("click", toggle_heatmap);