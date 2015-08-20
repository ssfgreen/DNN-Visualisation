
//----> The Visualisation Plot

var epochVis = function epochVis(s) {

  // initialise as a container with children
  BasicVis.Container.call(this, s);

  // appends a div to use later to print the info about the net (params etc)
  this.info_epoch = this.inner.append("div");
  this.info_vis = this.inner.append("div");

  // set tooltips function
  function tooltips (tooltype, appendage, labels) {
    // console.log("called");
    if (tooltype == "text") {
      console.log(tooltype);
      // creates a new tooltip, and binds to the new data!
      this.tooltext = new BasicVis.TextTooltip();
      this.tooltext._labels = labels;
      this.tooltext.bind(appendage.points);
      this.tooltext.bind_move(appendage.s);
      this.tooltext.div.style("font-size", "85%");
    } else if (tooltype == "img") {
      console.log(tooltype);
      this.toolimg = new BasicVis.ImgTooltip();
      this.toolimg.img_display.shape([28,28]);
      this.toolimg.img_display.imgs(mnist_xs);
      this.toolimg.bind(appendage.points);
      this.toolimg.bind_move(appendage.s);
    };
  }

  this.epoch_display = this.new_child(BasicVis.ScatterPlot)
    .N(DATA.EPOCH.length)
    .xrange.fit(DATA.EPOCH)
    .yrange.fit([1,2,3])
    .x(function(i) {return DATA.EPOCH[i];})
    .y(function(i) {return 2;})
    .color(function(i) {
      // set according to the label
      var hue = 180*DATA.EPOCH[i]/10.0;
      var saturation = 0.5;
      var lightness = 0.5;
      return d3.hsl(hue, saturation, lightness);
    });

  this.epoch_display
    .enable_zoom()
    .bindToWindowResize()

  this.main_display = this.new_child(BasicVis.ScatterPlot)
    .N(0)
    .color( function(i) {
      var hue = 180*DATA.TSNE_LABELS[i]/10.0;
      var saturation = 0.5;
      var lightness = 0.5;
      return d3.hsl(hue, saturation, lightness);
    });

  var main_display = this.main_display;

  this.main_display.show = function show(i) {
    var coords = DATA.TSNE_DATA[i];
    var labels = DATA.TSNE_LABELS;

    var NO_POINTS = 0;

    if (sensible_number_of_points < DATA.TSNE_DATA.length/2) {
      NO_POINTS = sensible_number_of_points;
    } else {
      NO_POINTS = DATA.TSNE_DATA[i].length/2;
    }

    main_display
      .N(NO_POINTS)
      .xrange.fit(coords)
      .yrange.fit(coords)
      .x(function(i) {return coords[2*i  ];}) // gets the x-coords responding to the index
      .y(function(i) {return coords[2*i+1];});

    main_display
    .enable_zoom()
    .bindToWindowResize()

    tooltips("text", main_display, labels);
  }

  // javascript scoping, to ensure it is THIS function
  // not the this of the 'circle' i.e
  var this_ = this;

  this.epoch_display.mouseover( function(i) {

    this_.info_epoch.html("<center> Layer: " + DATA.LAYER[i] + " Epoch: " + DATA.EPOCH[i]
        + " TO ADD MORE! </center");
    
    this_.main_display.show(i);
  });
}

epochVis.prototype = Object.create(BasicVis.Container.prototype);

epochVis.prototype.child_layout = function child_layout() {

  // gets the width of the main representation div 
  W = parseInt(this.s.style('width'));

  var gutter = W/20;
  var main = W - gutter*7;
  var box_size = main/2;
  var dims = [box_size, box_size];
  var H = W/2;

  this.inner
    .style('width', W)
    .style('height', H); 

  this.epoch_display.size(W/300);
  this.epoch_display.div
    // .style('border', '1px solid #959595')
    .style('background-color', 'white')
    .pos([gutter*3,gutter*3])
    .size(dims); 

  this.main_display.size(W/300);
  this.main_display.div
    // .style('border', '1px solid #959595')
    .style('background-color', 'white')
    .pos([box_size+gutter*4,gutter*3])
    .size(dims); 

  this.info_epoch
    .style('position', 'absolute')
    .style('font-size', '17px')
    .style('width', box_size)
    .style('height', box_size/3)
    .style('left', gutter*3)
    .style('top', gutter*4+box_size);

  return this;
}

function epoch_vis_space() {
  var epoch_visd = new epochVis("#epoch_vis");
  return epoch_visd;
}

epoch_vis_space();