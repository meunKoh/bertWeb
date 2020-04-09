$(document).ready(function() {
    var table = $('#mainTable').DataTable( {
        dom: 'Bfrtip',
        buttons: [{
            extend: 'excel',
            charset: 'UTF-8'
        }]
    } );
    table.buttons().container().appendTo( $('.col-sm-6:eq(-200)', table.table().container() ) );
} );
