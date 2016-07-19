// Configure requirejs
if (window.require) {
    window.require.config({
        map: {
            "*" : {
                "react": "https://fb.me/react-15.2.1.min.js",
                "react-dom": "https://fb.me/react-dom-15.2.1.min.js"
            }
        }
    });
}

import Manager from "./manager";

var handle_kernel = function(Jupyter, kernel) {
    if ( kernel.comm_manager && !kernel.component_manager ) {
      kernel.component_manager = new Manager( 'juno', kernel );
    }
};


function register_events(Jupyter, events) {
    // If a kernel already exists, create a widget manager.
    if (Jupyter.notebook && Jupyter.notebook.kernel) {
        handle_kernel(Jupyter, Jupyter.notebook.kernel);
    }
    // When the kernel is created, create a widget manager.
    events.on('kernel_created.Kernel kernel_created.Session', function(event, data) {
        handle_kernel(Jupyter, data.kernel);
    });
}



function add_css(url) {
  $('<link/>')
      .attr({
          rel: 'stylesheet',
          href: window.require.toUrl(url),
          type: 'text/css'
      })
      .appendTo('head');
}

function load_ipython_extension () {
    return new Promise(function(resolve) {
        requirejs([
            "base/js/namespace",
            "base/js/events",
            'react', 
            'react-dom'
        ], function( Jupyter, events, React, ReactDom ) {

            window.React = React;
            window.ReactDom = ReactDom;

            add_css('./nbextensions/juno_magic/juno.css');
            $('#kernel_indicator').append('<span id="juno_status"></span>');

            register_events(Jupyter, events);
            console.log("loaded juno");
            resolve();
        });
    });
}

module.exports = {
  load_ipython_extension: load_ipython_extension
};
