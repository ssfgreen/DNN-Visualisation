
  // t-SNE.js object and other global variables
  var step_counter = 0;
  var max_counter = 500;
  var dists; var all_labels; var svg; var timeout; var runner;
  var opt = {epsilon: 10};
  var tsne = new tsnejs.tSNE(opt);
  var color = d3.scale.category20();
  var final_dataset;

  // temporary variables here
  var key = "11sMddHxU5VdWvqGsCQ6QcHvo6HBl3xe-k-iMnQc5Hcw";
  // var key = null;
  var data = "data/CSV/mnist-small.csv";

  // code that is executed when page is loaded
  $(document).ready(function() {
    timeout = setTimeout(function() {
              document.getElementById("timeout_error").style.display = "inline";
                      }, 30000);
    if(key) {
       var list = document.getElementsByClassName("name_sheet");
       for(var i = 0; i < list.length; i++) list[i].style.display = "inline";
       Tabletop.init( { key: key,
                        // callback: init,
                        callback: showInfo,
                        simpleSheet: true,
                        parseNumbers: true } );
    }      
    else if(data) {
       var list = document.getElementsByClassName("name_csv")
       for(var i = 0; i < list.length; i++) list[i].style.display = "inline";
       var parsed_data = doParse(data);
       init(parsed_data, undefined);
    }
  });

  function showInfo(data, tabletop) {
    // alert("Successfully processed!")
    // console.log(data);
    init(data, undefined);
  }

  // function that returns the spreadsheet URL
  function openSheetUrl() {
    var win;
    if($_GET["key"].length == 44) win = window.open("https://docs.google.com/spreadsheets/d/" + $_GET["key"], "_blank");
    else win = window.open($_GET["key"], "_blank");
    win.focus();
  }

  // function that parses CSV data
  function doParse(data) {
    data = data.split('\n');
    var header = data[0].split(',');
    var parsed_data = new Array(data.length - 2);
    for(var i = 1; i < data.length - 1; i++) {
       parsed_data[i - 1] = new Array(header.length);
       var line = data[i].split(',');
       for(var j = 0; j < line.length; j++) {
          parsed_data[i - 1]["" + header[j]] = line[j];
       }
       parsed_data[i - 1]["rowNumber"] = i;
    }
    return parsed_data;
  }

  // function that executes after data is successfully loaded
  function init(data, tabletop) {
    clearTimeout(timeout);
    final_dataset = data;
    for(var i = 0; i < final_dataset.length; i++) final_dataset[i].focus = 0;
    dists = computeDistances(data);
    tsne.initDataDist(dists); 
    all_labels = new Array(data.length);
    for(var i = 0; i < data.length; i++) { all_labels[i] = data[i]["label"]; }
    drawEmbedding();
    runner = setInterval(step, 0);
  }

  // function that computes pairwise distances
  function computeDistances(data) {
    
    // initialize distance matrix
    var dist = new Array(data.length);
    for(var i = 0; i < data.length; i++) {
      dist[i] = new Array(data.length);
    }
    for(var i = 0; i < data.length; i++) {
      for(var j = 0; j < data.length; j++) {
        dist[i][j] = 0;
      }
    }

    // compute pairwise distances
    for(var i = 0; i < data.length; i++) {
      for(var j = i + 1; j < data.length; j++) {
        for(var d in data[0]) {
          if(d != "label" && d != "rowNumber" && d != focus) {
            dist[i][j] += Math.pow(data[i][d] - data[j][d], 2);
          }
        }
        dist[i][j] = Math.sqrt(dist[i][j]);
        dist[j][i] = dist[i][j];
      }
    }
    
    // normalize distances to prevent numerical issues
    var max_dist = 0;
    for(var i = 0; i < data.length; i++) {
      for(var j = i + 1; j < data.length; j++) {
        if(dist[i][j] > max_dist) max_dist = dist[i][j];
      }
    }
    for(var i = 0; i < data.length; i++) {
      for(var j = 0; j < data.length; j++) {
        dist[i][j] /= max_dist;
      }
    }
    return dist;
  }

  // function that changes the perplexity and restarts t-SNE
  function setPerplexity(p) {
    opt = {epsilon: p};
    tsne = new tsnejs.tSNE(opt);
    tsne.initDataDist(dists);
    step_counter = 0;
    clearInterval(runner);
    runner = setInterval(step, 0);
  }

  // function that updates embedding
  function updateEmbedding() {
    var Y = tsne.getSolution();
    svg.selectAll('.u')
       .attr("transform", function(d, i) { return "translate(" +
                                          ((Y[i][0] * 7 * ss + tx) + 450) + "," +
                                          ((Y[i][1] * 7 * ss + ty) + 300) + ")"; });
  }

  // function that draws initial embedding
  var div, tooltip;

  function drawEmbedding() {

    // Fill the embed div
    $("#laurens").empty();
    div = d3.select("#laurens");
    
    // Drawing area for map
    svg = div.append("svg")
     // .attr("style", "outline: 1px solid #333;")
     .attr("width", 900)
     .attr("height", 600)
     .style('background-color', 'white');

    // Retrieve all data
    var g = svg.selectAll(".b")
      .data(final_dataset)
      .enter().append("g")
      .attr("class", "u")

    // Add circle for each data point
    g.append("svg:circle")
     .attr("stroke-width", 1)
     .attr("fill",   function(d) { return d ? color(d["label"]) : "#00F"; })
     .attr("stroke", function(d) { return d ? color(d["label"]) : "#00F"; })
     .attr("fill-opacity", .65)
     .attr("stroke-opacity", .9)
     .attr("opacity", 1)
     .attr("class", "node1")
     .attr("r", 6)
     .attr("data-legend", function(d) { return d ? d["label"] : ""; })
     .on("mouseover", function(d) {
                          d.focus = 1;
                          tooltip.style("visibility", function(dd) { return dd.focus == 1 ? "visible" : "hidden"; })
                      })
     .on("mouseout", function(d) {
                          d.focus = 0;
                          tooltip.style("visibility", function(dd) { return dd.focus == 1 ? "visible" : "hidden"; })
                     });
      
     // Add tooltips
     tooltip = g.append("svg:text")
      .attr("dx", 0)
      .attr("dy", 0)
      .style("position", "absolute")
      .style("visibility", "hidden")
      .attr("text-anchor", "right")
      .style("font-size", "12px")
      .text(function(d) {
            // console.log(d);
            return d.label;

        });
      
    // Add zoom functionality to map
    var zoomListener = d3.behavior.zoom()
      .scaleExtent([0.1, 10])
      .center([0, 0])
      .on("zoom", zoomHandler);
    zoomListener(svg);

  }

  // function that handles zooming
  var tx = 0, ty = 0;
  var ss = 1;
  function zoomHandler() {
    tx = d3.event.translate[0];
    ty = d3.event.translate[1];
    ss = d3.event.scale;
    updateEmbedding();
  }

  // perform single t-SNE iteration
  function step() {
    step_counter++;
    if(step_counter <= max_counter) tsne.step();
    else {
        clearInterval(runner);
        // document.getElementById("exportButton").disabled = false;
    }
    updateEmbedding();
  }

