""" Adaptation of Laurens Van Der Maaten's tSNE 
    Animation using Karpathy's tSNE.js
"""

  // t-SNE.js object and other global variables
  var step_counter = 0;
  var max_counter = 500;
  var dists; var all_labels; var svg; var timeout; var runner;
  var opt = {epsilon: 10};
  var tsne = new tsnejs.tSNE(opt);
  var color = d3.scale.category20();
  var final_dataset;

  var str = localStorage.getItem("ngStorage-result");
  var coords = JSON.parse(str);
  console.log("data!!!: ", coords);

  var ID = coords.PARAMS.ID;
  var PARAMS = coords.PARAMS;
  var DATA = coords.DATA;
  var sensible_number_of_points = 500;

  // temporary variables for SHEETS access
  var key = "10-1alq2AZZd8njpAaQNufGh3_VXmFI7xvgTEEZGLzTQ"; // E20 L3
  // (NOTE - Have to allow google security to run it)


  // code that is executed when page is loaded & gets info from google
  $(document).ready(function() {
    timeout = setTimeout(function() {
              document.getElementById("timeout_error").style.display = "inline";
                      }, 3000);
    if(key) {
       var list = document.getElementsByClassName("name_sheet");
       for(var i = 0; i < list.length; i++) list[i].style.display = "inline";
       Tabletop.init( { key: key,
                        callback: showInfo,
                        simpleSheet: true,
                        parseNumbers: true } );
    }      
  });

  function showInfo(data, tabletop) {
    // alert("Successfully processed!")
    console.log(data);
    start(data, undefined);
  }

  // function that returns the spreadsheet URL
  function openSheetUrl() {
    var win;
    if($_GET["key"].length == 44) win = window.open("https://docs.google.com/spreadsheets/d/" + $_GET["key"], "_blank");
    else win = window.open($_GET["key"], "_blank");
    win.focus();
  }


  // function that executes after data is successfully loaded
  function start(data, tabletop) {
    console.log("init data", data)

    clearTimeout(timeout);

    final_dataset = data;

    for(var i = 0; i < final_dataset.length; i++) {
      final_dataset[i].focus = 0;
    }


    dists = computeDistances(data);

    tsne.initDataDist(dists); 

    all_labels = new Array(data.length);

    for(var i = 0; i < data.length; i++) {
       all_labels[i] = data[i]["label"];
        }

    render();
    runner = setInterval(step, 0);
  }

  // ----- IMPORTANT RENDERING!!
  var div, tooltip, ttbackground, pos;

  function render() {

    // Fill the embed div
    $("#laurens").empty();
    div = d3.select("#laurens");
    
    // Drawing area for map
    svg = div.append("svg")
     .attr("width", 900)
     .attr("height", 600)
     .style('background-color', 'rgba(119, 119, 119, 0.05)');

    // Retrieve all data
    var g = svg.selectAll(".b")
      .data(final_dataset)
      .enter().append("g")
      .attr("class", "u")

    // Add circle for each data point
    circles = g.append("circle")
       .attr("stroke-width", 1)
       .attr("fill",   function(d) { return d ? color(d["label"]) : "#00F"; })
       .attr("stroke", function(d) { return d ? color(d["label"]) : "#00F"; })
       .attr("fill-opacity", 0.9)
       .attr("stroke-opacity", 1.0)
       .attr("opacity", 1)
       .attr("class", "node1")
       .attr("r", 2.5)
       .on("mouseover", function(d) {
                            d.focus = 1;
                            var pos = [d3.event.pageX + 1000, d3.event.pageY + 4000 ];
                            bgmove(pos);
                            ttbackground.style("visibility", function(dd) { return dd.focus == 1 ? "visible" : "hidden"; })
                            tooltip.style("visibility", function(dd) { return dd.focus == 1 ? "visible" : "hidden"; })
                        })
       .on("mouseout", function(d) {
                            d.focus = 0;
                            bghide();
                            tooltip.style("visibility", function(dd) { return dd.focus == 1 ? "visible" : "hidden"; })
                       });
      

     // Add tooltips
     tooltip = g.append("text")
      .attr("dx", 0)
      .attr("dy", 0)
      .style("position", "absolute")
      .style("visibility", "hidden")
      .attr("text-anchor", "right")
      .style("font-size", "17px")
      .text(function(d) {
          return d.label;
        });


    // Add tooltips
   ttbackground = g.append("rect")
    .attr("width", 10)
    .attr("height", 1)
    .style("left", 0)
    .style("top", 0)
    .style("position", "absolute")
    .style("visibility", "hidden")
    .style('fill', 'black');

      
    // Add d3 zoom functionality to map
    var zoomListener = d3.behavior.zoom()
      .scaleExtent([0.1, 10])
      .center([0, 0])
      .on("zoom", zoomed);
    zoomListener(svg);

  function bgmove(val) {
    ttbackground
      .style("left", val[0])
      .style("top", val[1])
      // .style("visibility", "visible");
    return this;
  };

  function bghide() {
    ttbackground
      .style("visibility", "hidden");
    return this;
  };

  }

  // function that updates embedding
  function update() {
    var Y = tsne.getSolution();
    svg.selectAll('.u')
       .attr("transform", function(d, i) { return "translate(" +
                                          ((Y[i][0] * 7 * scals + tranx) + 450) + "," +
                                          ((Y[i][1] * 7 * scals + trany) + 300) + ")"; });
  }


  // ---- HELPERS

  // function that handles zooming
  var tranx = 0, trany = 0;
  var scals = 1;
  function zoomed() {
    tranx = d3.event.translate[0];
    trany = d3.event.translate[1];
    scals = d3.event.scale;
    update();
  }

  // perform single t-SNE iteration
  function step() {
    step_counter++;
    if(step_counter <= max_counter) tsne.step();
    else {
        clearInterval(runner);
    }
    update();
  }

  // function that computes pairwise distances (taken from Karpathy)
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