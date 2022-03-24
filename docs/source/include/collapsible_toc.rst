..
    CSS and javascript to enable collapsible table of contents. Adapted from
    scikit-learn's tune_toc.rst

.. raw:: html

   <style type="text/css">
     div[itemprop="articleBody"] li,
     div[itemprop="articleBody"] ul {
       transition-duration: 0.2s;
     }
     
     div[itemprop="articleBody"] li.toctree-l1 {
       background-color: #e5e5e5;
       list-style-type: none;
       margin-left: 0;
       margin-bottom: 1.2em;
       padding: 5px 0 0;
     }

     div[itemprop=articleBody] li.toctree-l1 > a {
       font-size: 110%;
       font-weight: bold;
       margin-left: 0.75rem;
     }
     
     div[itemprop="articleBody"] li.toctree-l2 {
       background-color: #ffffff;
       list-style-type: none !important;
       margin-left: 0;
       padding: 0.25em 0 0.25em 0 ;
     }
     
     div[itemprop="articleBody"] li.toctree-l2 ul {
       padding-left: 40px;
     }
     
     div[itemprop="articleBody"] li.toctree-l2:before {
       color: #777;
       content: attr(data-content);
       display: inline-block;
       font-size: 1rem;
       width: 1.5rem;
     }
     
     div[itemprop="articleBody"] li.toctree-l3 {
       list-style-type: square;
       margin-left: 0;
     }
     
     div[itemprop="articleBody"] li.toctree-l4 {
       list-style-type: circle !important;
       margin-left: 0;
     }
  </style>

   <script>
     window.addEventListener("DOMContentLoaded", function () {
       (function ($) {
         // Function to make the index toctree collapsible
         $(function () {
           $("div[itemprop=articleBody] .toctree-l2")
             .click(function (event) {
               if (event.target.tagName.toLowerCase() != "a") {
                 if ($(this).children("ul").length > 0) {
                   $(this).attr("data-content", !$(this).children("ul").is(":hidden") ? "▶" : "▼");
                   $(this).children("ul").toggle();
                 }
                 return true; // Makes links clickable
               }
             })
             .mousedown(function (event) {
               return false;
             }) // Firefox highlighting fix
             .children("ul")
             .hide();
           // Initialize the values
           $("div[itemprop=articleBody] li.toctree-l2:not(:has(ul))").attr("data-content", "-");
           $("div[itemprop=articleBody] li.toctree-l2:has(ul)").attr("data-content", "▶");
           $("div[itemprop=articleBody] li.toctree-l2:has(ul)").css("cursor", "pointer");
     
           $("div[itemprop=articleBody] .toctree-l2").hover(
             function () {
               if ($(this).children("ul").length > 0) {
                 $(this)
                   .css("background-color", "#f0f0f0")
                   .children("ul")
                   .css("background-color", "#f0f0f0");
                 $(this).attr("data-content", !$(this).children("ul").is(":hidden") ? "▼" : "▶");
               } else {
                 $(this).css("background-color", "#f2f2f2");
               }
             },
             function () {
               $(this)
                 .css("background-color", "white")
                 .children("ul")
                 .css("background-color", "white");
               if ($(this).children("ul").length > 0) {
                 $(this).attr("data-content", !$(this).children("ul").is(":hidden") ? "▼" : "▶");
               }
             }
           );
         });
       })(jQuery);
     });
   </script>
