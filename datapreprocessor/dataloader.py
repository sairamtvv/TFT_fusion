from pathlib import Path
import glob
import pathlib
import pandas as pd
import numpy as np

class  DataLoader:

    def __init__(self, fldr_pth: str, exclude_path_ls:list=[],  append_col:bool = True):
        """

        Args:
            fldr_pth: String  Absolute path to folde
            exclude_path_ls:
            append_col: Bool
        """
        self.fldr_pth = pathlib.Path(fldr_pth)
        self.exclude_path_ls = exclude_path_ls
        self.append_col = append_col




    def read_file(self, file_path):
        """


        Args:
            file_path:

        Returns:

        """
        # Get the file name without the extension
        file_name = file_path.split('/')[-1].split('.')[0]

        # Read the file into a data frame
        if file_path.endswith('.csv') or file_path.endswith('.CSV') or file_path.endswith('.dat'):

            #todo: this works only in linux
            if "Karna" in file_path.split("/")[-1]:
                df = pd.read_csv(file_path)
                sheet_names = "no sheet_name as it is csv"
                # Drop rows 1, 3, and 4
                rows_to_drop = [0, 1, 2, 4]  # List of row indices to drop
                df.drop(index=rows_to_drop, inplace=True)  # Drops multiple rows specified by the indices
                # Reset the header to the row of the DataFrame
                new_header = df.loc[3]  # the 3rd row contains the header
                df = df[1:]  # Remove the old header row
                df.columns = new_header  # Set the new header
                df.reset_index(drop=True, inplace=True)

            else:
                df = pd.read_csv(file_path, header=1)
                sheet_names = "no sheet_name as it is csv"
                # Drop rows 1, 3, and 4
                df = df.drop([0, 2, 3])
                # Reset the index
                df = df.reset_index(drop=True)
                df = df.drop([0])
                df = df.reset_index(drop=True)
            if np.nan in df.columns:
                df.drop(columns=[np.nan], inplace=True)
        elif file_path.endswith('.xlsx'):
            xl_file = pd.ExcelFile(file_path)
            sheet_names = xl_file.sheet_names
            if len(sheet_names) != 1:
                raise ValueError(
                    f"Excel file '{file_name}' has {len(sheet_names)} sheets, but should have exactly one sheet")
            sheet_name = sheet_names[0]
            df = pd.read_excel(xl_file, sheet_name=sheet_name)
            # Drop rows 1, 3, and 4
            df = df.drop([0, 2, 3])
            # Reset the index
            df = df.reset_index(drop=True)
            df = df.drop([0])
            df = df.reset_index(drop=True)
        else:
            raise ValueError(f"File '{file_name}' is not a CSV or Excel file")

        return df


    def _get_col_nm_values(self, file_pth, level_names_lst):
        """
        Naming Convention for the columns
        folder_0, folder_1, folder_2, folder_3, folder_4

        Args:
            filepath:

        Returns:

        """
        user_base_flder = self.fldr_pth.name
        folder_names = [folder.name for folder in file_pth.parents]
        # Find the index of the value in the list
        index = folder_names.index(user_base_flder)
        flder_parents = folder_names[0:index]


        if len(level_names_lst) == 0:
            level_names_lst = []
            for num, folder in enumerate(flder_parents):
                level_names_lst.append("folder_" + str(num))
        else:
            assert len(flder_parents) == len(level_names_lst), f"level_nameslist length should be" \
                                                           f"{len(flder_parents)} and they " \
                                                           f"are {flder_parents}"

        nm_value_lst = []
        for name, value in zip(level_names_lst, flder_parents):
            nm_value_lst.append((name, value))

        return nm_value_lst

    def fetch_df(self, level_names_lst=[]):

        # Use glob to retrieve the list of files
        file_paths = glob.glob(str(self.fldr_pth) + '/**/*', recursive=True)
        # Filter the list to select only file paths
        file_path_lst = [path for path in file_paths  if Path(path).is_file()]


        final_df = pd.DataFrame()
       # Print the list of files
        for file_path in file_path_lst:


            if file_path not in self.exclude_path_ls:
                df = self.read_file(file_path)
                if self.append_col:
                    nm_value_lst = self._get_col_nm_values(pathlib.Path(file_path), level_names_lst)
                    for name, value in nm_value_lst:
                        df[name] = value

                # if len(df.columns) !=8 and "Update_Offset" not in list(df.columns) :
                #print(file_path)
                try:
                    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
                except:
                    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], dayfirst=True)

                # Find duplicates and assign second occurrence with second value of 30
                duplicates = df.duplicated(subset='TIMESTAMP', keep='first')
                df.loc[duplicates, 'TIMESTAMP'] += pd.Timedelta(seconds=30)

                final_df = pd.concat([final_df, df])

        return final_df


if __name__ == "__main__":
    # flder_pth = "C:/Users/gurpreet.kaur/OneDrive - TruBoard Private Limited/Desktop/Raw_data/AP"
    # data_loader = DataLoader(flder_pth, append_col=False)
    # df = data_loader.fetch_df(level_names_lst=["year_month", "location"])
    # #df = data_loader.fetch_df()
    # a =1

    flder_pth = "C:/Users/gurpreet.kaur/OneDrive - TruBoard Private Limited/Desktop/Raw_data/AP/"
    data_loader = DataLoader(flder_pth, append_col=True)
    df = data_loader.fetch_df(level_names_lst=["year_month", "location"])
    #df = data_loader.fetch_df()
    a =1



